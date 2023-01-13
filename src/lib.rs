#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(unused_must_use)]
#![deny(unused_mut)]
// TODO: Ideally we could add some safety here to avoid errors where we are close to running out of
// disk space.
//
// TODO: Should probably have some linting mechanism here to ensure that we only use read_exact and
// write_all.
//
// TODO: Should add a switch to VersionedFile so that we can simulate a broken filesystem for ACID
// testing. Probably should have that switch accept a mutexed object which will tell it when to
// start failing. Then we can have many files at the same time coordinate various types of
// failures.
//
// TODO: We should add random occasional checks to the code which verify that the vf cursor matches
// the file handle cursor.
//
// TODO: Need to figure out how the versioned-file fits into a larger ACID ecosystem. Probably best
// approach is to make it a base layer that doesn't provide any acid guarantees (beyond the header,
// which is ACID compliant), and let larger transactional frameworks use versioned-file as a
// building block.

// NOTE: The way that we handle UpgradeFunc is unweildly and unfortunate. I was unable to figure
// out a clean way to do async function pointers in rust, and the method that I did find started to
// have issues as soon as one of the arguments was a pointer itself. If someone wants to take a
// stab at cleaning up the UpgradeFunc / WrappedUpgradeFunc structs and related code, improvements
// would be much welcomed.

//! The VersionedFile crate provides a wrapper to async_std::File which adds an invisible 4096 byte
//! header to the file which tracks things like the file version number and the file identifier.
//! When using methods like `seek` and `set_len`, the header will be ignored. For example, calling
//! set_len(0) will result in a file that has a logical size of 0 but the 4096 byte file header
//! will still be intact.
//!
//! The most useful portion of the header is the version number, which allows the software to
//! easily detect when a file is using an out-of-date version and complete an update on that file.
//! The other part of the header is an identifier which may help to recover the file in the event
//! that the file's name is changed or lost unexpectedly.
//!
//! When a VersionedFile is opened, the caller passes in the latest version of the file, along with
//! a chain of upgrades that can be used to upgrade older versions of the file to the most recent
//! version. Finally, the identifier is also supplied so that an error can be thrown if the file
//! has the wrong identifier.
//!
//! ```
//! // Basic file operations
//! use async_std::io::SeekFrom;
//! use std::path::PathBuf;
//!
//! use anyhow::{bail, Result, Error};
//! use versioned_file::{open_file, wrap_upgrade_process, Upgrade, VersionedFile};
//!
//! #[async_std::main]
//! async fn main() {
//!     // Create a file with open_file:
//!     let path = PathBuf::from("target/docs-example-file.txt");
//!     let identifier = "VersionedFileDocs::example.txt";
//!     let mut versioned_file = open_file(&path, identifier, 1, &Vec::new()).await.unwrap();
//!
//!     // Use write_all and read_exact for read/write operations:
//!     versioned_file.write_all(b"hello, world!").await.unwrap();
//!     versioned_file.seek(SeekFrom::Start(0)).await.unwrap();
//!     let mut buf = vec![0u8; versioned_file.len().await.unwrap() as usize];
//!     versioned_file.read_exact(&mut buf).await.unwrap();
//!     if buf != b"hello, world!" {
//!         panic!("example did not read correctly");
//!     }
//! }
//! ```
//! ```
//! // Simple upgrade example
//! use async_std::io::SeekFrom;
//! use std::path::PathBuf;
//!
//! use anyhow::{bail, Result, Error};
//! use versioned_file::{open_file, wrap_upgrade_process, Upgrade, VersionedFile};
//!
//! // An example of a function that upgrades a file from version 1 to version 2, while making
//! // changes to the body of the file.
//! async fn example_upgrade(
//!     mut vf: VersionedFile,
//!     initial_version: u8,
//!     updated_version: u8,
//! ) -> Result<(), Error> {
//!     // Check that the version is okay.
//!     if initial_version != 1 || updated_version != 2 {
//!         bail!("wrong version");
//!     }
//!
//!     // Truncate the file and replace the data
//!     vf.set_len(0).await.unwrap();
//!     let new_data = b"hello, update!";
//!     vf.write_all(new_data).await.unwrap();
//!     Ok(())
//! }
//!
//! #[async_std::main]
//! async fn main() {
//!     // Open a file with an upgrade process:
//!     let path = PathBuf::from("target/docs-example-file.txt");
//!     let identifier = "VersionedFileDocs::example.txt";
//!     let upgrade = Upgrade {
//!         initial_version: 1,
//!         updated_version: 2,
//!         process: wrap_upgrade_process(example_upgrade),
//!     };
//!     let mut vf = open_file(&path, identifier, 2, &vec![upgrade]).await.unwrap();
//!     let mut buf = vec![0u8; vf.len().await.unwrap() as usize];
//!     vf.read_exact(&mut buf).await.unwrap();
//!     if buf != b"hello, update!" {
//!         panic!("example did not update correctly");
//!     }
//!
//!     // Clean-up
//!     std::fs::remove_file(PathBuf::from("target/docs-example-file.txt"));
//! }

use async_std::fs::{File, OpenOptions};
use async_std::io::prelude::SeekExt;
use async_std::io::{ReadExt, SeekFrom, WriteExt};
use async_std::prelude::Future;
use std::collections::HashMap;
use std::path::PathBuf;
use std::pin::Pin;
use std::str::from_utf8;

use anyhow::{bail, Context, Error, Result};

/// UpgradeFunc is a pointer to a function that upgrades a file from one version to the next.
/// The intended starting and ending versions are explicitly stated in the input. When the upgrade
/// is complete, the file cursor will automatically be placed back at the start of the file.
///
/// It may seem rather redundant to explicitly declare the version transition in the input to the
/// UpgradeFunc, as the version numbers are already stated multiple times elsewhere as well.
/// Under normal circumstances, this redundancy would be seen as excessive, however a file upgrade
/// has the potential to corrupt or destory data, so we want extra layers of protection to ensure
/// that the wrong upgrade process is not called on a file.
///
/// This type is not usable until it has been wrapped with `wrap_upgrade_process`.

/// UpgradeFunc defines the signature for a function that can be used to upgrade a
/// VersionedFile. The UpgradeFunc function will receive the file that needs to be upgraded, and
/// it will also receive the intended initial version and upgraded version. The version inputs
/// allow the upgrade function to double check that the right upgrade is being used - if a bug in
/// the library somehow causes the wrong upgrade to be used, the user may end up with corrupted
/// data. For that reason, we place extra redundancy around the version checks.
///
/// UpgradeFunc functions cannot be used directly due to Rust's current inability to support
/// async function pointers. To use an UpgradeFunc, one must call `wrap_upgrade_process` first.
pub type UpgradeFunc =
    fn(file: VersionedFile, initial_version: u8, upgraded_version: u8) -> Result<(), Error>;

/// WrappedUpgradeFunc is a type that wraps an UpgradeFunc so that the UpgradeFunc can be
/// used as a function pointer in the call to `open_file`.
pub type WrappedUpgradeFunc =
    Box<dyn Fn(VersionedFile, u8, u8) -> Pin<Box<dyn Future<Output = Result<(), Error>>>>>;

/// wrap_upgrade_process is a function that will convert an UpgradeFunc into a
/// WrappedUpgradeFunc.
pub fn wrap_upgrade_process<T>(f: fn(VersionedFile, u8, u8) -> T) -> WrappedUpgradeFunc
where
    T: Future<Output = Result<(), Error>> + 'static,
{
    Box::new(move |x, y, z| Box::pin(f(x, y, z)))
}

/// Upgrade defines an upgrade process for upgrading the data in a file from one version to
/// another.
pub struct Upgrade {
    /// initial_version designates the version of the file that this upgrade should be applied to.
    pub initial_version: u8,
    /// updated_version designates the version of the file after the upgrade is complete.
    pub updated_version: u8,
    /// process defines the function that is used to upgrade the file.
    pub process: WrappedUpgradeFunc,
}

/// VersionedFile defines the main type for the crate, and implements an API for safely
/// manipulating versioned files. The API is based on the async_std::File interface, but with some
/// adjustments that are designed to make it both safer and more ergonomic. For example, len() is
/// exposed directly rather than having to first fetch the file metadata. Another example, all
/// calls to write will automatically flush() the file.
///
/// If a function is not fully documented, it is safe to assume that the function follows the same
/// convensions/rules as its equivalent function for async_std::File.
#[derive(Debug)]
pub struct VersionedFile {
    /// file houses the underlying file handle of the VersionedFile.
    file: File,

    /// cursor tracks the read offset of the underlying file. Implementers need to take care
    /// not to mix up the cursor as understood by the user with the cursor relative to
    /// the operating system.
    cursor: u64,

    /// needs_seek is set if the file needs to seek to synchronize with the cursor.
    needs_seek: bool,
}

// When working with the VersionedFile implementation, there are a few sharp corners to watch out
// for:
//
// The `cursor` field tracks the position of the cursor in the underlying file handle,
// and will therefore be 4096 larger than the values that the user is expecting when calling Seek.
//
// Every function must take care to properly update the cursor after completing its operation.
// In the event of an error, the cursor position of the file may be unknown. Every function that
// changes the file cursor position must ensure that in the event of a failure, `needs_seek` is set
// to `true`.
//
// Every function that depends on the file cursor must check `needs_seek` before executing, so that
// the cursor can be set to be equal to `cursor` in the event that the current position is
// unknown.
//
// Please test code careful when making changes.
impl VersionedFile {
    /// fix_seek will check if the file cursor has potentially drifted from the vf cursor and
    /// attempt to fix it if drift is possible.
    async fn fix_seek(&mut self) -> Result<(), Error> {
        // Seek to the correct location if the previous operation was not able to update the file
        // cursor.
        if self.needs_seek {
            match self.file.seek(SeekFrom::Start(self.cursor)).await {
                Ok(_) => self.needs_seek = false,
                Err(e) => bail!(format!(
                    "unable to set file cursor to correct position: {}",
                    e
                )),
            };
        }
        Ok(())
    }

    /// len will return the size of the file, not including the versioned header.
    pub async fn len(&mut self) -> Result<u64, Error> {
        let md = self
            .file
            .metadata()
            .await
            .context("unable to get metadata for file")?;
        Ok(md.len() - 4096)
    }

    /// read_exact will read from the data portion of a VersionedFile. If there is an error, the
    /// contents of buf are unspecified, and the read offset will not be updated.
    pub async fn read_exact(&mut self, buf: &mut [u8]) -> Result<(), Error> {
        self.fix_seek().await?;
        self.needs_seek = true;

        // Try reading the bytes. If that fails, try resetting the file cursor.
        match self.file.read_exact(buf).await {
            Ok(_) => {}
            Err(e) => {
                match self.file.seek(SeekFrom::Start(self.cursor)).await {
                    // If resetting the cursor is successful, we can clear the needs_seek flag
                    // before returning an error. Otherwise we leave the flag set so that future
                    // function calls know the cursor is in the wrong spot.
                    Ok(_) => self.needs_seek = false,
                    Err(_) => {}
                };
                bail!(format!("{}", e));
            }
        };

        // Update the cursor and clear the needs_seek flag.
        self.cursor += buf.len() as u64;
        self.needs_seek = false;
        Ok(())
    }

    /// seek will seek to the provided offset within the file, ignoring the header.
    pub async fn seek(&mut self, pos: SeekFrom) -> Result<u64, Error> {
        self.fix_seek().await?;
        self.needs_seek = true;

        // Special care needs to be taken with regards to the seeking boundaries. The udnerlying
        // file has boundaries that are the whole file, but the caller has boundaries that exclude
        // the first 4096 bytes.
        match pos {
            SeekFrom::Start(x) => {
                // If seeking from the start, we need to add 4096 bytes to the caller offset to
                // compensate for the header.
                let new_pos = self
                    .file
                    .seek(SeekFrom::Start(x + 4096))
                    .await
                    .context("versioned file seek failed")?;
                self.needs_seek = false;
                self.cursor = new_pos;
                return Ok(new_pos - 4096);
            }
            SeekFrom::End(x) => {
                // If seeking from the end, we need to make sure that the caller does not seek the
                // file into the header. We use self.len() because that will provide the length of
                // the file excluding the header.
                let size = self.len().await.context("unable to get file len")?;
                if x + (size as i64) < 0 {
                    self.needs_seek = false;
                    bail!("cannot seek to a position before the start of the file");
                }
                let new_pos = self.file.seek(pos).await.context("seek failed")?;
                self.needs_seek = false;
                self.cursor = new_pos;
                return Ok(new_pos - 4096);
            }
            SeekFrom::Current(x) => {
                if x + (self.cursor as i64) < 4096 {
                    self.needs_seek = false;
                    bail!("cannot seek to a position before the start of the file");
                }
                let new_pos = self.file.seek(pos).await.context("seek failed")?;
                self.needs_seek = false;
                self.cursor = new_pos;
                return Ok(new_pos - 4096);
            }
        }
    }

    /// set_len will truncate the file so that it has the provided length, excluding the header.
    /// This operation can be used to make the file larger as well. This operation will put the
    /// cursor at the end of the file after the length has been set.
    pub async fn set_len(&mut self, new_len: u64) -> Result<(), Error> {
        self.file
            .set_len(new_len + 4096)
            .await
            .context("unable to adjust file length")?;
        self.seek(SeekFrom::End(0))
            .await
            .context("unable to seek to new end of file")?;
        Ok(())
    }

    /// write_all will write to the VersionedFile and then call flush(). Note that flush() is not
    /// the same as fsync().
    ///
    /// If there is an error, the file cursor will not be updated.
    pub async fn write_all(&mut self, buf: &[u8]) -> Result<(), Error> {
        self.fix_seek().await?;
        self.needs_seek = true;

        // Try reading the bytes. If that fails, try resetting the file cursor.
        match self.file.write_all(buf).await {
            Ok(_) => {}
            Err(e) => {
                match self.file.seek(SeekFrom::Start(self.cursor)).await {
                    // If resetting the cursor is successful, we can clear the needs_seek flag
                    // before returning an error. Otherwise we leave the flag set so that future
                    // function calls know the cursor is in the wrong spot.
                    Ok(_) => self.needs_seek = false,
                    Err(_) => {}
                };
                bail!(format!("{}", e));
            }
        };

        // Flush the file. Note that this is not the same as fsync.
        self.file.flush().await.context("unable to flush file")?;

        // Update the cursor and clear the needs_seek flag.
        self.cursor += buf.len() as u64;
        self.needs_seek = false;
        Ok(())
    }
}

/// version_to_str will write out the version in ascii, adding leading zeroes if needed.
fn version_to_str(version: u8) -> Result<String, Error> {
    // 0 is not an allowed version, every other possible u8 is okay.
    if version == 0 {
        bail!("version is not allowed to be 0");
    }

    // Compute the 4 version bytes based on the latest version.
    let mut version_string = format!("{}", version);
    if version_string.len() == 1 {
        version_string = format!("00{}", version);
    } else if version_string.len() == 2 {
        version_string = format!("0{}", version);
    }
    Ok(version_string)
}

/// new_file_header will write the header of the file using the expected idenfitier and the latest
/// version.
async fn new_file_header(
    file: &mut File,
    expected_identifier: &str,
    latest_version: u8,
) -> Result<(), Error> {
    // Compute the full set of metadata bytes.
    let version_string =
        version_to_str(latest_version).context("unable to convert version to ascii string")?;
    let header_str = format!("{}\n{}\n", version_string, expected_identifier);
    let header_bytes = header_str.as_bytes();
    if header_bytes.len() > 256 {
        panic!("developer error: metadata_bytes should be guaranteed to have len below 256");
    }

    // Prepare the full header and write it to the file.
    let mut full_header = [0u8; 4096];
    full_header[..header_bytes.len()].copy_from_slice(header_bytes);
    file.write_all(&full_header)
        .await
        .context("unable to write initial metadata")?;
    file.flush()
        .await
        .context("unable to flush file after writing header")?;
    let new_metadata = file
        .metadata()
        .await
        .context("unable to get updated file metadata")?;
    if new_metadata.len() != 4096 {
        panic!(
            "developer error: file did not initialize with 4096 bytes: {}",
            new_metadata.len()
        );
    }

    // Reset the offset to 0; after creating the file the startup routine will read the data to
    // verify it matches.
    file.seek(SeekFrom::Start(0))
        .await
        .context("unable to seek back to beginning of file")?;

    Ok(())
}

/// verify_upgrade_paths verify that the set of paths provided for performing upgrades all lead to
/// the latest version, and will return an error if some path doesn't lead to the latest version.
/// It will also return an error if two possible paths exist for a given version.
fn verify_upgrade_paths(upgrade_paths: &Vec<Upgrade>, latest_version: u8) -> Result<(), Error> {
    // Enusre 0 was not used as the latest_version.
    if latest_version == 0 {
        bail!("version 0 is not allowed for a VersionedFile");
    }

    // Verify that an upgrade path exists for the file which carries it to the latest version.
    let mut version_routes = HashMap::new();
    // Verify basic properties of the graph (no cycles, no repeat sources).
    for path in upgrade_paths {
        if path.initial_version >= path.updated_version {
            bail!("upgrade paths must always lead to a higher version number");
        }
        if version_routes.contains_key(&path.initial_version) {
            bail!("upgrade paths can only have one upgrade for each version");
        }
        if path.updated_version > latest_version {
            bail!("upgrade paths lead beyond the latest version");
        }
        if path.initial_version == 0 {
            bail!("version 0 is not allowed for a VersionedFile");
        }
        version_routes.insert(path.initial_version, path.updated_version);
    }
    // Verify that all upgrades lead to the latest version. We iterate over the version_routes and mark every
    // node that connects to a finished node.
    let mut complete_paths = HashMap::new();
    complete_paths.insert(latest_version, {});
    loop {
        let mut progress = false;
        let mut finished = true;

        for (key, value) in &version_routes {
            if complete_paths.contains_key(key) {
                continue;
            }
            if complete_paths.contains_key(value) {
                progress = true;
                complete_paths.insert(*key, {});
            } else {
                finished = false;
            }
        }

        if finished {
            break;
        }
        if progress == false {
            bail!("update graph is incomplete, not all nodes lead to the latest version");
        }
    }

    Ok(())
}

/// perform_file_upgrade takes a file and an upgrade, and then executes the upgrade against the
/// file.
async fn perform_file_upgrade(filepath: &PathBuf, u: &Upgrade) -> Result<(), Error> {
    // Open the file and perform the upgrade.
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(filepath)
        .await
        .context("unable to open versioned file for update")?;
    let mut versioned_file = VersionedFile {
        file,
        cursor: 4096,
        needs_seek: false,
    };
    // Set the offset to right after the header, which is what the upgrade routines will expect.
    versioned_file
        .seek(SeekFrom::Start(0))
        .await
        .context("unable to seek in file after upgrade")?;
    (u.process)(versioned_file, u.initial_version, u.updated_version)
        .await
        .context(format!(
            "unable to complete file upgrade from version {} to {}",
            u.initial_version, u.updated_version
        ))?;
        // file drops automatically because it is moved into the path.process call.

    // Update the metadata to contain the new version string.
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(filepath)
        .await
        .context("unable to open versioned file for update")?;
    let mut versioned_file = VersionedFile {
        file,
        cursor: 4096,
        needs_seek: false,
    };
    let updated_version_str =
        version_to_str(u.updated_version).context("upgrade path has bad version")?;
    versioned_file
        .file
        .seek(SeekFrom::Start(0))
        .await
        .context("unable to seek to beginning of file")?;
    versioned_file
        .file
        .write_all(updated_version_str.as_bytes())
        .await
        .context("unable to write updated version to file header")?;

    Ok(())
}

/// open_file will open a versioned file.
///
/// If the file does not yet exist, a new VersionedFile will be created containing the
/// latest_version and the provided identifier in the header. If the file exists but is an older
/// version, the update_paths will be used to update the file to the latest version.
///
/// An error will be returned if the file does exist and has the wrong identifier, or if the file
/// has a version that is higher than 'latest_version', or if the upgrades do not provide a valid
/// path from the current version of the file to the latest version.
pub async fn open_file(
    filepath: &PathBuf,
    expected_identifier: &str,
    latest_version: u8,
    upgrades: &Vec<Upgrade>,
) -> Result<VersionedFile, Error> {
    // Verify that the inputs match all requirements.
    let path_str = filepath.to_str().context("could not stringify path")?;
    if !path_str.is_ascii() {
        bail!("path should be valid ascii");
    }
    if expected_identifier.len() > 251 {
        bail!("the identifier of a versioned file cannot exceed 251 bytes");
    }
    if !expected_identifier.is_ascii() {
        bail!("the identifier must be ascii");
    }

    // Open the file, creating a new file if one does not exist.
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(filepath)
        .await
        .context("unable to open versioned file")?;

    // If the file length is zero, we assume that the file has not been created yet.
    let file_metadata = file
        .metadata()
        .await
        .context("unable to read versioned file metadata")?;
    if file_metadata.len() == 0 {
        new_file_header(&mut file, expected_identifier, latest_version)
            .await
            .context("unable to write new file header")?;
    }

    // Read the first 4096 bytes of the file to get the file header.
    let mut header = vec![0; 4096];
    file.read_exact(&mut header)
        .await
        .context("unable to read file header")?;

    // Verify that the identifier bytes in the header match the expected identifier.
    let header_identifier = from_utf8(&header[4..4 + expected_identifier.len()])
        .context("the on-disk file identifier could not be parsed")?;
    if header_identifier != expected_identifier {
        bail!("the file does not have the correct identifier");
    }

    // Verify that the upgrade paths all lead to the latest version.
    verify_upgrade_paths(&upgrades, latest_version).context("upgrade paths are invalid")?;

    let version_str = from_utf8(&header[..3]).context("the on-disk version could not be parsed")?;
    let mut version: u8 = version_str
        .parse()
        .context("unable to parse on-disk version")?;

    // Had this weird issue with the async function pointer where I couldn't get it to work if I
    // made the VersionedFile a pointer. As a result, we have to pass the file in, which transfers
    // ownership, and means the file can't continue to be used in this function. I overcame that
    // limitation by just opening and closing the file repeatedly. This is the first such place
    // where we close the file for ownership reasons rather than because we don't need it anymore.
    drop(file);

    // Execute the upgrades.
    while version != latest_version {
        let mut found = false;
        for upgrade in upgrades {
            if upgrade.initial_version == version {
                perform_file_upgrade(filepath, upgrade)
                    .await
                    .context("unable to complete file upgrade")?;
                version = upgrade.updated_version;
                found = true;
                break;
            }
        }

        // The upgrades verifier ensures that if an upgrade exists in the set of upgrades, then
        // there also exists a path to the latest_version from that upgrade. Therefore, if this
        // file doesn't have a path to the latest version, no other upgrades will be executed
        // either.
        if !found {
            bail!("no viable upgrade path exists for file");
        }
    }

    // Open the file and create the versioned_file.
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(filepath)
        .await
        .context("unable to open versioned file for update")?;
    file.seek(SeekFrom::Start(4096))
        .await
        .context("unable to seek to beginning of file after header")?;
    let versioned_file = VersionedFile {
        file,
        cursor: 4096,
        needs_seek: false,
    };

    Ok(versioned_file)
}

#[cfg(test)]
mod tests {
    use super::*;

    use testdir::testdir;

    // Create a helper function which does a null upgrade so that we can do testing of the upgrade
    // path verifier.
    async fn stub_upgrade(_: VersionedFile, _: u8, _: u8) -> Result<(), Error> {
        Ok(())
    }

    // This is a basic upgrade function that expects the current contents of the file to be
    // "test_data". It will alter the contents so that they say "test".
    async fn smoke_upgrade_1_2(
        mut vf: VersionedFile,
        initial_version: u8,
        updated_version: u8,
    ) -> Result<(), Error> {
        // Verify that the correct version is being used.
        if initial_version != 1 || updated_version != 2 {
            bail!("this upgrade is intended to take the file from version 1 to version 2");
        }
        if vf.len().await.unwrap() != 9 {
            bail!("file is wrong len");
        }
        // Read the file and verify that we are upgrading the correct data.
        let mut buf = [0u8; 9];
        vf.read_exact(&mut buf)
            .await
            .context("unable to read old file contents")?;
        if &buf != b"test_data" {
            bail!(format!("file appears corrupt: {:?}", buf));
        }

        // Truncate the file and write the new data into it.
        let new_data = b"test";
        vf.set_len(0).await.unwrap();
        vf.write_all(new_data)
            .await
            .context("unable to write new data after deleting old data")?;
        Ok(())
    }

    // smoke upgrade 2->3
    async fn smoke_upgrade_2_3(
        mut vf: VersionedFile,
        initial_version: u8,
        updated_version: u8,
    ) -> Result<(), Error> {
        // Verify that the correct version is being used.
        if initial_version != 2 || updated_version != 3 {
            bail!("this upgrade is intended to take the file from version 2 to version 3");
        }
        if vf.len().await.unwrap() != 4 {
            bail!("file is wrong len");
        }
        // Read the file and verify that we are upgrading the correct data.
        let mut buf = [0u8; 4];
        vf.read_exact(&mut buf)
            .await
            .context("unable to read old file contents")?;
        if &buf != b"test" {
            bail!("file appears corrupt");
        }

        // Truncate the file and write the new data into it.
        let new_data = b"testtest";
        vf.set_len(0).await.unwrap();
        vf.write_all(new_data)
            .await
            .context("unable to write new data after deleting old data")?;
        Ok(())
    }

    // smoke upgrade 3->4
    async fn smoke_upgrade_3_4(
        mut vf: VersionedFile,
        initial_version: u8,
        updated_version: u8,
    ) -> Result<(), Error> {
        // Verify that the correct version is being used.
        if initial_version != 3 || updated_version != 4 {
            bail!("this upgrade is intended to take the file from version 1 to version 2");
        }
        if vf.len().await.unwrap() != 8 {
            bail!("file is wrong len");
        }
        // Read the file and verify that we are upgrading the correct data.
        let mut buf = [0u8; 8];
        vf.read_exact(&mut buf)
            .await
            .context("unable to read old file contents")?;
        if &buf != b"testtest" {
            bail!("file appears corrupt");
        }

        // Truncate the file and write the new data into it.
        let new_data = b"testtesttest";
        vf.set_len(0).await.unwrap();
        vf.write_all(new_data)
            .await
            .context("unable to write new data after deleting old data")?;
        Ok(())
    }

    #[async_std::test]
    // Do basic testing of all the major functions for VersionedFiles
    async fn smoke_test() {
        // Create a basic versioned file.
        let dir = testdir!();
        let test_dat = dir.join("test.dat");
        open_file(&test_dat, "versioned_file::test.dat", 0, &Vec::new())
            .await
            .context("unable to create versioned file")
            .unwrap_err();
        open_file(&test_dat, "versioned_file::test.dat", 1, &Vec::new())
            .await
            .context("unable to create versioned file")
            .unwrap();
        // Try to open it again.
        open_file(&test_dat, "versioned_file::test.dat", 1, &Vec::new())
            .await
            .context("unable to create versioned file")
            .unwrap();
        // Try to open it with the wrong specifier.
        open_file(&test_dat, "bad_versioned_file::test.dat", 1, &Vec::new())
            .await
            .context("unable to create versioned file")
            .unwrap_err();

        // Try to make some invalid new files.
        let invalid_name = dir.join("❄️"); // snowflake emoji in filename
        open_file(&invalid_name, "versioned_file::test.dat", 1, &Vec::new())
            .await
            .context("unable to create versioned file")
            .unwrap_err();
        let invalid_id = dir.join("invalid_identifier.dat");
        open_file(&invalid_id, "versioned_file::test.dat::❄️", 1, &Vec::new())
            .await
            .context("unable to create versioned file")
            .unwrap_err();

        // Perform a test where we open test.dat and write a small amount of data to it. Then we
        // will open the file again and read back that data.
        let mut file = open_file(&test_dat, "versioned_file::test.dat", 1, &Vec::new())
            .await
            .unwrap();
        file.write_all(b"test_data").await.unwrap();
        let mut file = open_file(&test_dat, "versioned_file::test.dat", 1, &Vec::new())
            .await
            .unwrap();
        if file.len().await.unwrap() != 9 {
            panic!("file has unexpected len");
        }
        let mut buf = [0u8; 9];
        file.read_exact(&mut buf).await.unwrap();
        if &buf != b"test_data" {
            panic!("data read does not match data written");
        }
        // Try to open the file again and ensure the write happened in the correct spot.
        open_file(&test_dat, "versioned_file::test.dat", 1, &Vec::new())
            .await
            .unwrap();

        // Open the file again, this time with an upgrade for smoke_upgrade_1_2.
        let mut upgrade_chain = vec![Upgrade {
            initial_version: 1,
            updated_version: 2,
            process: wrap_upgrade_process(smoke_upgrade_1_2),
        }];
        let mut file = open_file(&test_dat, "versioned_file::test.dat", 2, &upgrade_chain)
            .await
            .unwrap();
        if file.len().await.unwrap() != 4 {
            panic!("file has wrong len");
        }
        let mut buf = [0u8; 4];
        file.read_exact(&mut buf).await.unwrap();
        if &buf != b"test" {
            panic!("data read does not match data written");
        }
        // Try to open the file again to make sure everything still completes.
        open_file(&test_dat, "versioned_file::test.dat", 2, &upgrade_chain)
            .await
            .unwrap();

        // Attempt to do two upgrades at once, from 2 to 3  and 3 to 4.
        upgrade_chain.push(Upgrade {
            initial_version: 2,
            updated_version: 3,
            process: wrap_upgrade_process(smoke_upgrade_2_3),
        });
        upgrade_chain.push(Upgrade {
            initial_version: 3,
            updated_version: 4,
            process: wrap_upgrade_process(smoke_upgrade_3_4),
        });
        let mut file = open_file(&test_dat, "versioned_file::test.dat", 4, &upgrade_chain)
            .await
            .unwrap();
        if file.len().await.unwrap() != 12 {
            panic!("file has wrong len");
        }
        let mut buf = [0u8; 12];
        file.read_exact(&mut buf).await.unwrap();
        if &buf != b"testtesttest" {
            panic!("data read does not match data written");
        }
        // Try to open the file again to make sure everything still completes.
        let mut file = open_file(&test_dat, "versioned_file::test.dat", 4, &upgrade_chain)
            .await
            .unwrap();

        // Test that the seeking is implemented correctly.
        file.seek(SeekFrom::End(-5)).await.unwrap();
        file.write_all(b"NOVELLA").await.unwrap();
        file.seek(SeekFrom::Current(-3)).await.unwrap();
        file.seek(SeekFrom::Current(-4)).await.unwrap();
        file.seek(SeekFrom::Current(-7)).await.unwrap();
        let mut buf = [0u8; 14];
        file.read_exact(&mut buf).await.unwrap();
        if &buf != b"testtesNOVELLA" {
            panic!(
                "read data has unexpected result: {} || {}",
                std::str::from_utf8(&buf).unwrap(),
                buf[0]
            );
        }
        file.seek(SeekFrom::Current(-2)).await.unwrap();
        file.seek(SeekFrom::End(-15)).await.unwrap_err();
        let mut buf = [0u8; 2];
        file.read_exact(&mut buf).await.unwrap();
        if &buf != b"LA" {
            panic!("seek_end error changed file cursor");
        }
        file.seek(SeekFrom::Current(-2)).await.unwrap();
        file.seek(SeekFrom::Current(-13)).await.unwrap_err();
        file.read_exact(&mut buf).await.unwrap();
        if &buf != b"LA" {
            panic!("seek_end error changed file cursor");
        }
    }

    #[test]
    // Attempt to provide comprehensive test coverage of the upgrade path verifier.
    fn test_verify_upgrade_paths() {
        // Passing in no upgrades should be fine.
        verify_upgrade_paths(&Vec::new(), 0).unwrap_err(); // 0 is not a legal version
        verify_upgrade_paths(&Vec::new(), 1).unwrap();
        verify_upgrade_paths(&Vec::new(), 2).unwrap();
        verify_upgrade_paths(&Vec::new(), 255).unwrap();

        // Passing in a single upgrade should be okay.
        verify_upgrade_paths(
            &vec![Upgrade {
                initial_version: 1,
                updated_version: 2,
                process: wrap_upgrade_process(stub_upgrade),
            }],
            2,
        )
        .unwrap();

        // A non-increasing upgrade is not okay.
        verify_upgrade_paths(
            &vec![Upgrade {
                initial_version: 2,
                updated_version: 2,
                process: wrap_upgrade_process(stub_upgrade),
            }],
            2,
        )
        .unwrap_err();

        // No route to final version is not okay.
        verify_upgrade_paths(
            &vec![Upgrade {
                initial_version: 1,
                updated_version: 2,
                process: wrap_upgrade_process(stub_upgrade),
            }],
            3,
        )
        .unwrap_err();

        // Simple path is okay.
        verify_upgrade_paths(
            &vec![
                Upgrade {
                    initial_version: 1,
                    updated_version: 2,
                    process: wrap_upgrade_process(stub_upgrade),
                },
                Upgrade {
                    initial_version: 2,
                    updated_version: 3,
                    process: wrap_upgrade_process(stub_upgrade),
                },
            ],
            3,
        )
        .unwrap();

        // Two starting options for the same version is not okay.
        verify_upgrade_paths(
            &vec![
                Upgrade {
                    initial_version: 1,
                    updated_version: 2,
                    process: wrap_upgrade_process(stub_upgrade),
                },
                Upgrade {
                    initial_version: 2,
                    updated_version: 3,
                    process: wrap_upgrade_process(stub_upgrade),
                },
                Upgrade {
                    initial_version: 1,
                    updated_version: 3,
                    process: wrap_upgrade_process(stub_upgrade),
                },
            ],
            3,
        )
        .unwrap_err();

        // Two ending options for the same version is okay.
        verify_upgrade_paths(
            &vec![
                Upgrade {
                    initial_version: 1,
                    updated_version: 3,
                    process: wrap_upgrade_process(stub_upgrade),
                },
                Upgrade {
                    initial_version: 2,
                    updated_version: 3,
                    process: wrap_upgrade_process(stub_upgrade),
                },
            ],
            3,
        )
        .unwrap();

        // Two ending options for the same version, version too high.
        verify_upgrade_paths(
            &vec![
                Upgrade {
                    initial_version: 1,
                    updated_version: 3,
                    process: wrap_upgrade_process(stub_upgrade),
                },
                Upgrade {
                    initial_version: 2,
                    updated_version: 3,
                    process: wrap_upgrade_process(stub_upgrade),
                },
            ],
            2,
        )
        .unwrap_err();

        // Complex valid structure.
        verify_upgrade_paths(
            &vec![
                Upgrade {
                    initial_version: 1,
                    updated_version: 3,
                    process: wrap_upgrade_process(stub_upgrade),
                },
                Upgrade {
                    initial_version: 2,
                    updated_version: 3,
                    process: wrap_upgrade_process(stub_upgrade),
                },
                Upgrade {
                    initial_version: 3,
                    updated_version: 6,
                    process: wrap_upgrade_process(stub_upgrade),
                },
                Upgrade {
                    initial_version: 4,
                    updated_version: 6,
                    process: wrap_upgrade_process(stub_upgrade),
                },
                Upgrade {
                    initial_version: 5,
                    updated_version: 6,
                    process: wrap_upgrade_process(stub_upgrade),
                },
            ],
            6,
        )
        .unwrap();

        // Complex valid structure, randomly ordered.
        verify_upgrade_paths(
            &vec![
                Upgrade {
                    initial_version: 5,
                    updated_version: 6,
                    process: wrap_upgrade_process(stub_upgrade),
                },
                Upgrade {
                    initial_version: 2,
                    updated_version: 3,
                    process: wrap_upgrade_process(stub_upgrade),
                },
                Upgrade {
                    initial_version: 3,
                    updated_version: 6,
                    process: wrap_upgrade_process(stub_upgrade),
                },
                Upgrade {
                    initial_version: 1,
                    updated_version: 3,
                    process: wrap_upgrade_process(stub_upgrade),
                },
                Upgrade {
                    initial_version: 4,
                    updated_version: 6,
                    process: wrap_upgrade_process(stub_upgrade),
                },
            ],
            6,
        )
        .unwrap();

        // Complex structure, randomly ordered, one orphan.
        verify_upgrade_paths(
            &vec![
                Upgrade {
                    initial_version: 2,
                    updated_version: 5,
                    process: wrap_upgrade_process(stub_upgrade),
                },
                Upgrade {
                    initial_version: 6,
                    updated_version: 7,
                    process: wrap_upgrade_process(stub_upgrade),
                },
                Upgrade {
                    initial_version: 3,
                    updated_version: 6,
                    process: wrap_upgrade_process(stub_upgrade),
                },
                Upgrade {
                    initial_version: 1,
                    updated_version: 4,
                    process: wrap_upgrade_process(stub_upgrade),
                },
                Upgrade {
                    initial_version: 4,
                    updated_version: 6,
                    process: wrap_upgrade_process(stub_upgrade),
                },
            ],
            6,
        )
        .unwrap_err();
    }

    #[test]
    fn test_version_to_str() {
        version_to_str(0).unwrap_err();
        if version_to_str(1).unwrap() != "001" {
            panic!("1 failed");
        }
        if version_to_str(2).unwrap() != "002" {
            panic!("2 failed");
        }
        if version_to_str(9).unwrap() != "009" {
            panic!("9 failed");
        }
        if version_to_str(39).unwrap() != "039" {
            panic!("39 failed");
        }
        if version_to_str(139).unwrap() != "139" {
            panic!("139 failed");
        }
    }
}
