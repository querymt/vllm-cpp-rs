use std::fs;
use std::path::{Path, PathBuf};

const LIB_DIR_CANDIDATES: [&str; 2] = ["lib64", "lib"];

pub(crate) fn shared_library_name(stem: &str) -> String {
    format!("lib{stem}.so")
}

pub(crate) fn require_library_file(
    directory: &Path,
    expected_artifact: &str,
    description: &str,
) -> Result<PathBuf, String> {
    let path = directory.join(expected_artifact);
    if path.is_file() {
        Ok(path)
    } else {
        Err(format!(
            "expected {description} artifact {expected_artifact} at {}; install or copy the artifact there",
            path.display()
        ))
    }
}

pub(crate) fn find_installed_library_dir(
    root: &Path,
    expected_artifact: &str,
) -> Result<PathBuf, String> {
    let searched_dirs: Vec<PathBuf> = LIB_DIR_CANDIDATES
        .iter()
        .map(|name| root.join(name))
        .collect();
    let mut matches: Vec<(PathBuf, PathBuf)> = Vec::new();

    for directory in &searched_dirs {
        let candidate = directory.join(expected_artifact);
        if candidate.is_file() {
            let canonical = fs::canonicalize(&candidate).unwrap_or_else(|_| candidate.clone());
            if !matches.iter().any(|(_, existing)| existing == &canonical) {
                matches.push((directory.clone(), canonical));
            }
        }
    }

    match matches.as_slice() {
        [(directory, _)] => Ok(directory.clone()),
        [] => Err(format!(
            "expected {expected_artifact} below {} in exactly one standard library directory; searched {}",
            root.display(),
            searched_dirs
                .iter()
                .map(|path| path.join(expected_artifact).display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )),
        _ => Err(format!(
            "found ambiguous {expected_artifact} artifacts below {} in {}; set VLLM_CPP_LIB_DIR to the intended directory",
            root.display(),
            matches
                .iter()
                .map(|(path, _)| path.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )),
    }
}

pub(crate) fn cmake_cache_path(cache: &Path, key: &str) -> Result<PathBuf, String> {
    let contents = fs::read_to_string(cache)
        .map_err(|error| format!("could not read {}: {error}", cache.display()))?;
    let prefix = format!("{key}:");
    let matches: Vec<&str> = contents
        .lines()
        .filter_map(|line| Some(line.strip_prefix(&prefix)?.split_once('=')?.1))
        .collect();

    let value = match matches.as_slice() {
        [value] => *value,
        [] => return Err(format!("{key} is absent from {}", cache.display())),
        _ => {
            return Err(format!(
                "{key} occurs more than once in {}",
                cache.display()
            ));
        }
    };
    if value.is_empty() {
        return Err(format!("{key} is empty in {}", cache.display()));
    }
    if value.ends_with("-NOTFOUND") {
        return Err(format!(
            "{key} is unresolved ({value}) in {}",
            cache.display()
        ));
    }

    let path = PathBuf::from(value);
    if !path.is_absolute() {
        return Err(format!(
            "{key} must be an absolute path in {}; got {value:?}",
            cache.display()
        ));
    }
    Ok(path)
}
