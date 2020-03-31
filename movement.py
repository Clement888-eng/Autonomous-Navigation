def movebot():
    # publish to cmd_vel to move TurtleBot
    pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)

    twist = Twist()
    twist.linear.x = linear_speed
    twist.angular.z = 0.0
    time.sleep(1)
    pub.publish(twist)

def turn_right():
    rospy.loginfo(['Start turning right !!!' , get_direction()])
    rotatebot(90)

def turn_left():
    rospy.loginfo(['Start turning left !!!' , get_direction()])
    rotatebot(-90)

def move_backward():
    pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
    twist = Twist()
    rospy.loginfo(['Start moving backward!!!' , get_direction()])
    rotatebot(180)
    twist.linear.x = linear_speed
    twist.angular.z = 0.0
    time.sleep(1)
    pub.publish(twist)

def is_left_free():
    if laser_range[270] > stop_distance:
        return True
    else:
        return False

def is_forward_free():
    if laser_range[0] > stop_distance:
        return True
    else:
        return False

def start():
    if is_left_free() == True:
        #rotate 90 deg to left
        #move forward
        #run start() again
    else:
        if is_forward_free() == True:
            #move forward
            #run start() again
        else:
            #rotate 90 deg to right
            #run start() again

def move(distance, accuracy):
    start=round(laser_range[0],2)
    end=round(laser_range[0],2)-distance
    while abs(laser_range[0]-end)>=accuracy:
        movebot()
    stopbot()

def start():
    if is_left_free() == True:
        #rotate 90 deg to left
        turn_left()
        #move forward
        move(stop_distance,accuracy)
        #run start() again
        start()
    else:
        if is_forward_free() == True:
            #move forward
            move(stop_distace,accuracy)
            #run start() again
            start()
        else:
            #rotate 90 deg to right
            turn_right()
            #run start() again
            start()


def forward():
    #check is there any unmapped area infront
    #return True is yes
    #return False is no

def move():
    if forward()==False:
        #try rotate robot 90 deg to right
        move()
    else:
        #check any unmapped area right/left to the front
        if True:
            #move forward X cm turn right/left and move forward Y cm
            move()
        else:
            #map is completed
    
        
        
dy=laser_range[0]/resolution

def forward_right():
    lst=[]
    for i in range(dy):
        if im2arrr[height//2-i][width//2]==1:
            lst.append(im2arr[height//2-i][width:])
    return lst

        
def forward_left():
	lst=[]
	for i in range(height//2):
		if im2arr[height//2-i][width//2]==1:
			lst.append(im2arr[height//2-i][width//2::-1])
		if im2arr[height//2-i][width//2]>1:
			break
	for j in range(len(lst)):
		for x in range(len(lst[0])):
			if lst[j][x]==1 and lst[j][x+1]==0:
				return ['L', j*resolution, (x+1)*resolution]
				break
			elif lst[j][x]==1 and lst[j][x+1]>0:
				continue
		break
	return False

def forward_right():
    j=0
    lst=[]
    for i in range(height//2):
	    if im2arr[height//2-i][width//2]==1:
		    lst.append(im2arr[height//2-i][width//2::])
	    if im2arr[height//2-i][width//2]>1:
		    break
    for j in range(len(lst)):
	    for x in range(len(lst[0])):
		    if lst[j][x]==1 and lst[j][x+1]==0:
			    return ['R',j*resolution,(x+1)*resolution]
			    break
		    elif lst[j][x]==1 and lst[j][x+1]>0:
			    continue
	    break
    return False

def move():
	right=forward_right()
	left=forward_left()
	if forward_right()==False and forward_left()==False:
		print(rotatebot(90))
	else:
		if right[1]+right[2]<=left[1]+left[2]:
			return forward_right()
		else:
			return forward_left()

    
    
