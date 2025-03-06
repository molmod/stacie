#!/usr/bin/env bash
rsync -avR --info=progress2 $1:projects/emd-viscosity/stacie/* .
