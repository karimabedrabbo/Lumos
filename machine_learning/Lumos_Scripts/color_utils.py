

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def rgb2hsv(r, g, b):
	# H value is [0, 360]. S, V values are [0, 1].
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = df/mx
    v = mx

    # change HSV values to match openCV. H value is [0, 180]. S, V values are [0, 255].
    h = h / 2.0
    s = s * 255.0
    v = v * 255.0
    
    return (int(h), int(s), int(v))

# create a +/- range for each hsv color that is input

"""
Standard Deviation:

A value lower than 25 means that it's a low-contrast image whose colors don't vary much. 
This means that the HSV range should be lower.

A value between 25 and 70 means that it's a mid-contrast image whose colors vary enough. 
This means that the HSV range should be moderate.

A value higher than 75 means that it's a high-contrast image whose colors vary a LOT. 
This means that the HSV range should be higher.

"""


#
#
#
# low-contrast
#
#
#

def hsv2above_low(h, s, v):
	h_above = h + 2
	if s >= 240:
		s_above = 255
	else:
		s_above = s + 15

	if v >= 240:
		v_above = 255
	else:
		v_above = v + 15
	return (h_above, s_above, v_above)

def hsv2below_low(h, s, v):
	h_below = h - 2
	s_below = s - 15
	v_below = v - 15
	return (h_below, s_below, v_below)


def hsv2above_cup_low(h, s, v):
	h_above = h + 1
	if s >= 245:
		s_above = 255
	else:
		s_above = s + 10

	if v >= 245:
		v_above = 255
	else:
		v_above = v + 10
	return (h_above, s_above, v_above)

def hsv2below_cup_low(h, s, v):
	h_below = h - 1
	s_below = s - 10
	v_below = v - 10
	return (h_below, s_below, v_below)

#
#
#
# mid-contrast
#
#
#

def hsv2above(h, s, v):
	h_above = h + 2
	if s >= 235:
		s_above = 255
	else:
		s_above = s + 20

	if v >= 235:
		v_above = 255
	else:
		v_above = v + 20
	return (h_above, s_above, v_above)

def hsv2below(h, s, v):
	h_below = h - 2
	s_below = s - 20
	v_below = v - 20
	return (h_below, s_below, v_below)


def hsv2above_cup(h, s, v):
	h_above = h + 1
	if s >= 245:
		s_above = 255
	else:
		s_above = s + 10

	if v >= 245:
		v_above = 255
	else:
		v_above = v + 10
	return (h_above, s_above, v_above)

def hsv2below_cup(h, s, v):
	h_below = h - 1
	s_below = s - 10
	v_below = v - 10
	return (h_below, s_below, v_below)

#
#
#
# mid-contrast
#
#
#


def hsv2above_high(h, s, v):
	h_above = h + 3
	if s >= 225:
		s_above = 255
	else:
		s_above = s + 30

	if v >= 225:
		v_above = 255
	else:
		v_above = v + 30
	return (h_above, s_above, v_above)

def hsv2below_high(h, s, v):
	h_below = h - 3
	s_below = s - 30
	v_below = v - 30
	return (h_below, s_below, v_below)


def hsv2above_cup_high(h, s, v):
	h_above = h + 2
	if s >= 235:
		s_above = 255
	else:
		s_above = s + 20

	if v >= 235:
		v_above = 255
	else:
		v_above = v + 20
	return (h_above, s_above, v_above)

def hsv2below_cup_high(h, s, v):
	h_below = h - 2
	s_below = s - 20
	v_below = v - 20
	return (h_below, s_below, v_below)