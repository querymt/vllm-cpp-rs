{
  description = "Rust bindings for vllm.cpp";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = inputs:
    inputs.flake-parts.lib.mkFlake {inherit inputs;} {
      systems = inputs.nixpkgs.lib.systems.flakeExposed;

      perSystem = {system, ...}: let
        overlays = [inputs.rust-overlay.overlays.default];
        pkgs = import inputs.nixpkgs {
          inherit system overlays;
        };

        rustToolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
      in {
        devShells.default = pkgs.mkShell {
          packages = [
            rustToolchain
            pkgs.cmake
            pkgs.ninja
            pkgs.pkg-config
            pkgs.llvmPackages.clang
            pkgs.llvmPackages.bintools
            pkgs.rust-bindgen
          ];

          shellHook = ''
            export PS1="(dev:vllm-cpp-rs) $PS1"
          '';
        };
      };
    };
}
