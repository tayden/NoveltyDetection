ls ./*.ipynb | xargs -n 1 -I {} sh -c "jupyter nbconvert --to python {}"

