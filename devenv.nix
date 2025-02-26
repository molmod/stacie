{ pkgs, lib, config, inputs, ... }:

{
  # https://devenv.sh/packages/
  packages = with pkgs; [
    # Packages with binaries: take from nix instead of pip
    (python3.withPackages (ps: with ps; [
      matplotlib
      numpy
      msgpack
    ]))
  ];

  # https://devenv.sh/languages/
  # languages.rust.enable = true;
  languages.python = {
    enable = true;
    venv.enable = true;
    venv.requirements = "-e .[docs,tests]";
  };
}
