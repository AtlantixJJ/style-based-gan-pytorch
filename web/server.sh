#tar cvfz - * | ssh -p 8081 atlantix@166.111.17.31 "cd disk6/GANDebugServer; tar xvfz -"
tar cvfz - * | ssh sr "cd srgan; tar xvfz -"