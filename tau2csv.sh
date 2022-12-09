# find out how many functions are 
# profiled by TAU: this will be
# the number of lines we need to
# pull from the end of pprof's
# output
numfuns=$(pprof -l | tail +2 | wc -l)

# convert the lines of functions
# into a comma-separated list
# of functions
funlist=$(pprof -l \
            | tail +2 \
            | tail -$numfuns \
            | sed -E 's/\(.*\)//g' \
            | paste -s -d,)

# convert the 3rd column of the
# pprof mean data into a comma-
# separated list of msec values
echo $(pprof -s \
        | tail -$numfuns \
        | awk -F ' ' '{print $3}' \
        | sed -E 's/,//g' \
        | paste -s -d,)

