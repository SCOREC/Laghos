#!/bin/bash -ex
d=/users/kamrak/MFEM-Laghos/mfem_build_noOpt_debug/mfem
make MFEM_DIR=$d \
LAGHOS_DEBUG=YES \
CONFIG_MK=$d/share/mfem/config.mk \
TEST_MK=$d/share/mfem/test.mk $1
