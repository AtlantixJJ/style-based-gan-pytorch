tar cvfz - script | ssh jr "cd data/srgan/; tar xvfz -"
tar cvfz - *.py train model script figure | ssh img "cd srgan/; tar xvfz -"
#tar cvfz - train model script *.py | ssh fit "cd disk6/srgan/; tar xvfz -"
