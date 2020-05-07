tar cvfz - *.py script | ssh img "cd srgan/; tar xvfz -"
tar cvfz - *.py script | ssh jr "cd data/srgan/; tar xvfz -"
#tar cvfz - train model script *.py | ssh fit "cd disk6/srgan/; tar xvfz -"
