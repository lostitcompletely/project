RATIOS:
Forumlas used to find ratios
(landmark1-landmark2)/(landmark3-landmark4) for non empty columns in data/ratios.csv 
used in the face.distances method

PFL-PFH = (||33-133|| + ||362-263||) / (||159-145|| + ||386-374||)
eyebrow = 2 * ||223-159|| / ||159-145||
golden = ||2-200|| / ||9-2||
fifths = ||234-454|| / 5
midface = 2 * ||468-473|| / (||468-0|| + ||473-0||)

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

ANGLES:
Forumlas used to find ANGLES
arccos((|L1-L2|.|L3-L2|) / sqrt(||L1-L2||^2+||L3-L2||^2)) angle between landmark1, landmark2 and landmark3 
where landmark 2 is the point the 2 vectors meet in data/angles.csv
used in the face.theta method

facial convexity angle (FCA) = averages the left and right yaw angles to get the facial convexity angle using the method for other angles

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

SYMMETRIES:
Finding a way to measure symmetries 
found the centre line of the face using set landmarks [61, 146, 91, 181, 13, 311, 321, 375, 291] or [10, 151, 6, 4, 2, 13, 14, 152]
reflected points about that line and found the RSS of the reflected points and the points already on the face 
(see the self.symmetry_measurements dictionary in the initialization of the face class)
then scaled the RSS to be between 0 and 1