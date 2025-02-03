cd docs
./make html
cd ..
python -m build
twine upload dist/*
git add –-update
git push -u origin master
