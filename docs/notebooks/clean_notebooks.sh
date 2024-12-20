#! /bin/bash
# Remove all the output from the jupyter notebooks


for x in $(ls molecule.ipynb);
do
    echo ${x}
    jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace "${x}"
    jupyter nbconvert --to notebook --execute --inplace "${x}"
done

# Clean up temporal files
# git clean -fdx