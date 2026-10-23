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
        cudaPkgs = import inputs.nixpkgs {
          inherit system overlays;
          config.allowUnfree = true;
        };

        rustToolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
        msrvToolchain = pkgs.rust-bin.stable."1.85.0".default.override {
          extensions = ["clippy" "rustfmt"];
        };
      in {
        devShells =
          {
            default = pkgs.mkShell {
              packages = [
                rustToolchain
                pkgs.cmake
                pkgs.git
                pkgs.just
                pkgs.jq
                pkgs.ninja
                pkgs.pkg-config
                pkgs.python3
                pkgs.gnutar
                pkgs.llvmPackages.clang
                pkgs.llvmPackages.bintools
                pkgs.rust-bindgen
              ];

              shellHook = ''
                export PS1="(dev:vllm-cpp-rs) $PS1"
              '';
            };

            msrv = pkgs.mkShell {
              packages = [
                msrvToolchain
                pkgs.cmake
                pkgs.just
                pkgs.ninja
                pkgs.pkg-config
                pkgs.llvmPackages.clang
                pkgs.llvmPackages.bintools
              ];

              shellHook = ''
                export PS1="(msrv:vllm-cpp-rs) $PS1"
              '';
            };
          }
          // pkgs.lib.optionalAttrs
          (builtins.elem system [
            "x86_64-linux"
            "aarch64-linux"
          ]) {
            cuda = let
              toolkit = cudaPkgs.cudaPackages.cudatoolkit;
              cutlass = cudaPkgs.cudaPackages.cutlass;
            in
              pkgs.mkShell {
                packages = [
                  rustToolchain
                  pkgs.cmake
                  pkgs.git
                  pkgs.just
                  pkgs.jq
                  pkgs.ninja
                  pkgs.pkg-config
                  pkgs.gnutar
                  pkgs.llvmPackages.clang
                  pkgs.llvmPackages.bintools
                  pkgs.rust-bindgen
                  toolkit
                  cutlass
                ];

                shellHook = ''
                  export PS1="(cuda:vllm-cpp-rs) $PS1"
                  export CUDA_PATH="${toolkit}"
                  export CUDA_HOME="$CUDA_PATH"
                  export CUDAToolkit_ROOT="$CUDA_PATH"
                  export VLLM_CPP_CUTLASS_DIR="${cutlass.src}"
                  if [ -d /run/opengl-driver/lib ]; then
                    export LD_LIBRARY_PATH="/run/opengl-driver/lib:${toolkit}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
                  else
                    export LD_LIBRARY_PATH="${toolkit}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
                  fi
                '';
              };

            vulkan = let
              vulkanLibraryPath = pkgs.lib.makeLibraryPath [
                pkgs.vulkan-loader
                pkgs.vulkan-validation-layers
              ];
            in
              pkgs.mkShell {
                packages = [
                  rustToolchain
                  pkgs.cmake
                  pkgs.git
                  pkgs.just
                  pkgs.jq
                  pkgs.ninja
                  pkgs.pkg-config
                  pkgs.gnutar
                  pkgs.llvmPackages.clang
                  pkgs.llvmPackages.bintools
                  pkgs.rust-bindgen
                  pkgs.vulkan-tools
                  pkgs.vulkan-loader
                  pkgs.vulkan-validation-layers
                ];

                shellHook = ''
                  export PS1="(vulkan:vllm-cpp-rs) $PS1"
                  if [ -d /run/opengl-driver/lib ]; then
                    export LD_LIBRARY_PATH="/run/opengl-driver/lib:${vulkanLibraryPath}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
                  else
                    export LD_LIBRARY_PATH="${vulkanLibraryPath}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
                  fi
                  export VK_LAYER_PATH="${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d"
                '';
              };
          };
      };
    };
}
