#tar cvfz - train model script *.py | ssh jr "cd data/srgan/; tar xvfz -"
tar cvfz - model script *.py | ssh img "cd srgan/; tar xvfz -"
#tar cvfz - train model script *.py | ssh fit "cd disk6/srgan/; tar xvfz -"