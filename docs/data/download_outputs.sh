#!/usr/bin/env bash
rsync -av --info=progress2 $1:projects/emd-viscosity/stacie/* .
