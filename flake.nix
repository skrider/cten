{
    description = "CUDA tensor library";
    inputs = {
        flake-utils.url = "github:numtide/flake-utils";
        nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
        cutlass = {
            url = "github:NVIDIA/cutlass/v3.4.0";
            flake = false;
        };
        cuda-samples = {
            url = "github:NVIDIA/cuda-samples/v12.3";
            flake = false;
        };
    };
    outputs = inputs:
    let 
        system = "x86_64-linux";
        pkgs = import inputs.nixpkgs {
            inherit system;
            config.allowUnfree = true;
        };
        updateVendor = pkgs.writeShellScriptBin "update-vendor" ''
        rm -rf vendor
        mkdir -p vendor/cuda
        ln -s ${inputs.cutlass} vendor/cutlass
        ln -s ${pkgs.cudaPackages_12_3.cudatoolkit}/include vendor/cuda/include
        ln -s ${inputs.cuda-samples} vendor/cuda_samples
        '';
    in {
        devShells.${system}.default = pkgs.mkShell {
            nativeBuildInputs = with pkgs; [
                cudaPackages_12_3.cudatoolkit
                gcc12
                gnumake
                updateVendor
            ];
        };
    };
}