# Representing numbers with variable names for convenience in reference.
normal, train, val, test, train_val, ig = range(6)

################################################################################
##  Configuration section. Modify everything below                            ##
################################################################################
shuffle = True
srcpath = 'raw'
destpath = 'example'
pfx = 'pi'
constructs = {
  "dare": train,
  "fail": val,
  "forget": test,
  "have chance": train,
}

#################################################################################
### This part contains some post processing code needed for the module import
lookup = [(.8,.1,.1), (1.,0.,0.), (0.,1.,0.), (0.,0.,1.), (.9,.1,0.), (0.,0.,0.)]
splits = {srcpath+'/'+'_'.join(x.split()+["data"])+'.txt' : lookup[constructs[x]]\
          if constructs[x] in range(6) else constructs[x] for x in constructs}
