{ pkgs, lib, config, inputs, ... }:

{
  # https://devenv.sh/packages/
  # See https://github.com/cachix/devenv/issues/1264
  packages = with pkgs; [
    stdenv.cc.cc.lib # required by jupyter
    gcc-unwrapped # fix: libstdc++.so.6: cannot open shared object file
    libz # fix: for numpy/pandas import
  ];

  # https://devenv.sh/languages/
  # languages.rust.enable = true;
  languages.python = {
    enable = true;
    venv.enable = true;
    venv.requirements = ''
      ipykernel
      matplotlib
      numpy
      msgpack
      -e .[docs,tests]
    '';
  };

  env.LD_LIBRARY_PATH = "${pkgs.gcc-unwrapped.lib}/lib64:${pkgs.libz}/lib";
}
