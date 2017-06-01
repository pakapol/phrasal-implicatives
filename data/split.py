import os, sys, random
from itertools import product, compress

def match_identity(id1, id2):
  return (id1[0] in [id2[0],id2[2]] or id1[2] in [id2[0],id2[2]]) and (id1[1] in [id2[1],id2[3]] or id1[1] in [id2[1],id2[3]])

def match_previous_identities(iden, previous):
  total = 0
  for ids in previous:
    if not match_identity(iden, ids):
      total += 1
  if 0.9 < float(total) / len(previous) and len(previous) > 4:
    return False
  return True

def extract_data(fname, splits):
  constr_name = " ".join(os.path.splitext(os.path.basename(fname))[0].split("_")[:-1])
  train_p, val_p, test_p = splits
  all_data = []
  with open(fname, 'r') as f:
    line_count = 0

    # Set the identification of each line
    # For the beginning, the line identity is a list of 4 blank strings
    prev_line_identities = [["","","",""]] * 100

    # Keep track of what examples have been added to the block
    # The current block is still empty
    current_block = []
    curr_line_identity = []
    token = []
    for line in f:
      if line_count % 4 == 0:
        token.append(line[:-1])
        prem = line[:-1].split(" ")
        curr_line_identity.extend([prem[0], prem[-1]])
      if line_count % 4 == 1:
        token.append(line[:-1])
      if line_count % 4 == 2:
        token.append(line[:-1])
        hyp = line[:-1].split(" ")
        curr_line_identity.extend([hyp[0], hyp[-1]])
      if line_count % 4 == 3:
        token.append(constr_name)
        if match_previous_identities(curr_line_identity, prev_line_identities):
          current_block.append(tuple(token))
          if len(prev_line_identities) < 100:
            prev_line_identities.append(curr_line_identity)
        else:
          if current_block:
            all_data.append(current_block)
          current_block = [tuple(token)]
          prev_line_identities = [curr_line_identity]
        token = []
        curr_line_identity = []
      line_count += 1
  random.shuffle(all_data)

  train = []
  val = []
  test = []
  block_count = 0

  for blocks in all_data:
    if block_count + 0.5 <= test_p * len(all_data):
       test.extend(blocks)
    elif block_count + 0.5 <= (val_p + test_p) * len(all_data):
       val.extend(blocks)
    elif block_count + 0.5 <= (val_p + test_p + train_p) * len(all_data):
       train.extend(blocks)
    block_count += 1
  print("From \"{}\", sorted into Train: {}, Val: {}, Test {}".format(constr_name, len(train),len(val),len(test)))
  return train, val, test

def correct_formats(fname):
  if os.path.splitext(fname)[1] != ".txt":
    return False
  try:
    f = open(fname, 'r')
    line_count = 0
    for line in f:
      if line_count % 4 == 1 and line[:-1] not in ['entails', 'contradicts','permits']:
        return False
      if line_count % 4 == 3 and len(line[:-1].strip(" ")) > 0:
        return False
      line_count += 1
    f.close()
  except IOError:
    return False
  return True

def interactive_session(splits=None, shuffle=None, srcpath=None, destpath=None, pfx=None):
  """
  Run a sequence of interactive session by prompting users for file paths
  and confirm each source files status that are ready for reading step
  Need to return: splits, shuffle, destpath, pfx

  splits: A dictionary with mapping of filepaths -> split ratio for that source file
      e.g. {'data/forget_data.txt':(1.0,0.0,0.0)}
  shuffle: A boolean, indicating whether we shuffle the training output
  destpath: Path where destination files reside
  pfx: Prefix of destination files. Default to 'pi'
  """
  #########################################################################
  # Section 1: Prompt user for the directory and retrieve list of filenames
  # Return: srcpath, files_list
  while 1:
    try:
      if srcpath is None:
        srcpath = raw_input("Enter path to files: ")
      files_list = os.listdir(srcpath)
      break
    except OSError:
      print("Invalid directory. Please try again")
      srcpath = None
  ###########################################################################
  # Section 2: Prompt for destination directory
  # Return: destpath, dest_list
  while 1:
    try:
      if destpath is None:
        destpath = raw_input("Enter destination path: ")
      dest_list = os.listdir(destpath)
      break
    except OSError:
      print("Directory not found. Please try again")
      destpath = None
  print("The files will be written to {}".format(\
         os.path.join(destpath,"pi.[prem,label,hyp,constr].[train,val,test]")))
  ############################################################################
  # Section 3: Prompt user for prefix of the destination files names
  # Return: pfx
  while 1:
    if pfx is None:
      pfx = raw_input("To change the prefix, enter a new prefix name here: ")
    if pfx == "":
      pfx = "pi"
    possible_names = ["{}.{}.{}".format(pfx,i,j) for i,j in\
                      product(["prem","label","hyp","constr"],["train","val","test"])]
    for fn in dest_list:
      if fn in possible_names:
        print("{} already exists! Change your destination name".format(fn))
        pfx = None
        break
    else:
      break

  ############################################################################
  # Section 4: Filter only text files of the right format
  # check formats for each file and list out the file status
  # Return: files_list, files_valid
  print("Filtering only .txt files and checking format ... ")
  files_list = map(lambda x: os.path.join(srcpath, x), files_list)
  files_valid = map(correct_formats, files_list)
  print("Files status:")
  for i in range(len(files_list)):
    print("  OK  {}".format(files_list[i]) if files_valid[i] else "  IGN {}".format(files_list[i]))

  ##############################################################################
  # Section 5 user direction: specify train/val/test ratio for each source files
  print("For each files, please specify the ratio to be splitted into the each split.")
  print("  0: Standard   80:10:10 (to train:val:test) Press enter to take this option.")
  print("  1: Train only 100:0:0")
  print("  2: Val only   0:100:0")
  print("  3: Test only  0:0:100")
  print("  4: Train+Val  90:10:0")
  print("  5: Ignore     0:0:0")
  print("  For other ratios, enter <num>:<num>:<num>. The sum of numbers should be between 0 to 100")

  #############################################################################
  # Actual Section 5: specify train/val/test ratio for each source files
  # For each file, ask for the destination
  # Return: splits
  if splits is None:
    splits = {}
    lookup = {"":(.8,.1,.1), "0":(.8,.1,.1), "1":(1.,0.,0.), "2":(0.,1.,0.),\
              "3":(0.,0.,1.), "4":(.9,.1,0.), "5":(0.,0.,0.)}
    for i in range(len(files_list)):
      name = files_list[i]
      if files_valid[i]:
        while 1:
          inp = raw_input(name + " : ").replace(" ","")
          if inp in lookup:
            break
          elif len(inp.split(":")) == 3:
            try:
              l = map(float, inp.split(":"))
              if 0. <= sum(l) <= 100.:
                break
            except ValueError:
              pass
          print("Invalid, retry")
        if inp in lookup:
          splits[name] = lookup[inp]
        else:
          splits[name] = tuple(map(lambda x: float(x)/100, inp.split(":")))
  else:
    names = list(compress(files_list, files_valid))
    keys = splits.keys()
    for sname in keys:
      if sname not in names:
        print("Warning: construct \"{}\" not found, ignoring".format(sname))
        del splits[sname]    
  #############################################################################
  # Warn about block split
  print("")
  print("NOTE: This splitting script will split the data by BLOCK,")
  print("  which uses some heuristics for block boundary detection.")
  print("  The ratio specified above will indicate the ratio of the")
  print("  number of blocks assigned to each split.")
  print("")

  #############################################################################
  # Actual Section 6: Prompt if user want to shuffle
  # Return: shuffle
  if shuffle is None:
    while 1:
      shf_prompt = raw_input("Do you want to shuffle the data on the output end? (Enter for yes) : ")
      if shf_prompt.lower() in ["yes","y","no","n",""]:
        shuffle = shf_prompt in ["yes","y",""]
        break
      print("Invalid, retry with yes or no")
    print("")

  ### Finished! Return stuff!
  return splits, shuffle, srcpath, destpath, pfx

def get_config(fname):
  try:
    fname = os.path.splitext(fname)[0]
    exec("import {}".format(fname))
    exec("shuffle = {}.shuffle".format(fname))
    exec("srcpath = {}.srcpath".format(fname))
    exec("destpath = {}.destpath".format(fname))
    exec("pfx = {}.pfx".format(fname))
    exec("splits = {}.splits".format(fname))
  except ImportError:
    print("Invalid file name or file not found.")
    sys.exit()
  return splits, shuffle, srcpath, destpath, pfx

if __name__ == '__main__':
  random.seed("PI")
  if len(sys.argv) > 2:
    print("Usage: python split.py")
    print("       python split.py <config-file>")
    sys.exit()
  if len(sys.argv) == 2:
    splits, shuffle, srcpath, destpath, pfx = get_config(sys.argv[1])
    splits, shuffle, srcpath, destpath, pfx = interactive_session(splits=splits, shuffle=shuffle, srcpath=srcpath, destpath=destpath, pfx=pfx)
  else:
    splits, shuffle, srcpath, destpath, pfx = interactive_session()

  ############################ REAL STUFF BELOW ################################
  # variable from above: splits, shuffle, destpath, pfx
  # splits: Dictionary from path -> splits
  # shuffle: shuffle outputs?
  # destpath, pfx: destination filenames

  # Deploy block detection heuristics
  # Block capping (None/number)
  # Random ordering in
  # Random ordering out
  train = []
  val = []
  test = []
  for fname in splits:
    train_add, val_add, test_add = extract_data(fname, splits[fname])
    train.extend(train_add)
    val.extend(val_add)
    test.extend(test_add)
  if shuffle:
    random.shuffle(train)
  print("In total, Train: {}, Val: {}, Test {}".format(len(train),len(val),len(test)))

  # Name: pi.prem.train pi.hyp.train pi.label.train pi.constr.train
  # Name: pi.prem.val pi.hyp.val pi.label.val pi.constr.val
  # Name: pi.prem.test pi.hyp.test pi.label.test pi.constr.test

  to_write = {}
  if train:
    to_write[".prem.train"], to_write[".label.train"], to_write[".hyp.train"], to_write[".constr.train"] = zip(*train)
  if val:
    to_write[".prem.val"], to_write[".label.val"], to_write[".hyp.val"], to_write[".constr.val"] = zip(*val)
  if test:
    to_write[".prem.test"], to_write[".label.test"], to_write[".hyp.test"], to_write[".constr.test"] = zip(*test)

  for sfx in to_write:
    try:
      os.remove(os.path.join(destpath, pfx+sfx))
    except OSError:
      # There is absolutely no need to handle this error. Only used as a control flow.
      pass

    with open(os.path.join(destpath, pfx+sfx), 'w') as f:
      for token in to_write[sfx]:
        f.write(token + '\n')
