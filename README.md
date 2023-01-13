# versioned-file

This readme is copy-pasted from the rust docs, and may be out of date.

The VersionedFile crate provides a wrapper to async_std::File which adds an invisible 4096 byte
header to the file which tracks things like the file version number and the file identifier.
When using methods like `seek` and `set_len`, the header will be ignored. For example, calling
set_len(0) will result in a file that has a logical size of 0 but the 4096 byte file header
will still be intact.

The most useful portion of the header is the version number, which allows the software to
easily detect when a file is using an out-of-date version and complete an update on that file.
The other part of the header is an identifier which may help to recover the file in the event
that the file's name is changed or lost unexpectedly.

When a VersionedFile is opened, the caller passes in the latest version of the file, along with
a chain of upgrades that can be used to upgrade older versions of the file to the most recent
version. Finally, the identifier is also supplied so that an error can be thrown if the file
has the wrong identifier.

```rs
// Basic file operations
use async_std::io::SeekFrom;
use std::path::PathBuf;

use anyhow::{bail, Result, Error};
use versioned_file::{open_file, wrap_upgrade_process, Upgrade, VersionedFile};

#[async_std::main]
async fn main() {
    // Create a file with open_file:
    let path = PathBuf::from("target/docs-example-file.txt");
    let identifier = "VersionedFileDocs::example.txt";
    let mut versioned_file = open_file(&path, identifier, 1, &Vec::new()).await.unwrap();

    // Use write_all and read_exact for read/write operations:
    versioned_file.write_all(b"hello, world!").await.unwrap();
    versioned_file.seek(SeekFrom::Start(0)).await.unwrap();
    let mut buf = vec![0u8; versioned_file.len().await.unwrap() as usize];
    versioned_file.read_exact(&mut buf).await.unwrap();
    if buf != b"hello, world!" {
        panic!("example did not read correctly");
    }
}
```
```rs
// Simple upgrade example
use async_std::io::SeekFrom;
use std::path::PathBuf;

use anyhow::{bail, Result, Error};
use versioned_file::{open_file, wrap_upgrade_process, Upgrade, VersionedFile};

// An example of a function that upgrades a file from version 1 to version 2, while making
// changes to the body of the file.
async fn example_upgrade(
    mut vf: VersionedFile,
    initial_version: u8,
    updated_version: u8,
) -> Result<(), Error> {
    // Check that the version is okay.
    if initial_version != 1 || updated_version != 2 {
        bail!("wrong version");
    }

    // Truncate the file and replace the data
    vf.set_len(0).await.unwrap();
    let new_data = b"hello, update!";
    vf.write_all(new_data).await.unwrap();
    Ok(())
}

#[async_std::main]
async fn main() {
    // Open a file with an upgrade process:
    let path = PathBuf::from("target/docs-example-file.txt");
    let identifier = "VersionedFileDocs::example.txt";
    let upgrade = Upgrade {
        initial_version: 1,
        updated_version: 2,
        process: wrap_upgrade_process(example_upgrade),
    };
    let mut vf = open_file(&path, identifier, 2, &vec![upgrade]).await.unwrap();
    let mut buf = vec![0u8; vf.len().await.unwrap() as usize];
    vf.read_exact(&mut buf).await.unwrap();
    if buf != b"hello, update!" {
        panic!("example did not update correctly");
    }

    // Clean-up
    std::fs::remove_file(PathBuf::from("target/docs-example-file.txt"));
}
```
