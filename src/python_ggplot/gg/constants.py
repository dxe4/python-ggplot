# this disables something that has to be fiexed
# need to start gathering those types of issues here so we can fix faster
DISABLE_MODIFY_FOR_STACKING = True

# Originally stat identity was created with assumption that X and Y is given
# which is sensible for identity BUT some geoms can take in
# (x="'', ymin='', ymax='')
# some temporary work arounds are in place but they cause more fundamental issues
# need to provide an abstract way of dealing this, some work is already planned
USE_Y_X_MINMAX_AS_X_VALUES = True
