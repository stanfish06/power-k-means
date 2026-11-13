{
  description = "power-k-means dev shell: C++ (CMake, Boost) and Python binding (Cython, uv)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";

  outputs =
    { self, nixpkgs }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          packages = [
            # C++: stdenv provides the wrapped compiler (sets CXX); boost190
            # provides BoostConfig.cmake, so CMake skips the header download
            pkgs.cmake
            pkgs.ninja
            pkgs.boost190
            pkgs.clang-tools # clang-format, clangd

            # Python: interpreter CMake finds for the manual extension build
            (pkgs.python313.withPackages (ps: [
              ps.numpy
              ps.cython
              ps.pytest
            ]))
            pkgs.uv
            pkgs.ruff
            pkgs.lefthook
          ];
        };
      });
    };
}
