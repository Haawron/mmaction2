#!/bin/bash

echo -e "Testing vanillas...\n"
. slurm/test_by_jids.sh {16847..16852}

echo -e "\n\n\n\n"

echo -e "Testing danns...\n"
. slurm/test_by_jids.sh {16904..16909}

echo -e "\n\n\n\n"

echo -e "Testing osbps...\n"
. slurm/test_by_jids.sh {16910..16915}
