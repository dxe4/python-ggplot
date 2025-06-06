# this disables something that has to be fiexed
# need to start gathering those types of issues here so we can fix faster
DISABLE_MODIFY_FOR_STACKING = True
# Originally stat identity was created with assumption that X and Y is given
# which is sensible for identity BUT some geoms can take in
# (x="'', ymin='', ymax='')
# some temporary work arounds are in place but they cause more fundamental issues
# need to provide an abstract way of dealing this, some work is already planned
USE_Y_X_MINMAX_AS_X_VALUES = True
# this has to be fixed before making an alpha release
# managed to get away with this so far
SKIP_APPLY_TRANSOFRMATIONS = True
# This flips x and y because the original code expected X to be mandatory
# but later on it became optional
# this is needs fixing before alpha, but fine for noe
FLIP_COLUMNS_IF_NEEDED = True

# for XYScaleComposite and XYScalePrimaryComposite
# we set by default the primary scale to be X
# what happens if you do coord_flip ?
DEFAULT_TO_X = True
