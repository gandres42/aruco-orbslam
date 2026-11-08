## Usage
### Re-scaling
If ORB-SLAM updates scale after 

### Re-anchoring
If the ORB-SLAM map is re-scaled after scale recovery the map placement may be incorrect.  Call the following to re-estimate origin based on an anchor tag.
```
ros2 service call /aruco/reanchor std_srvs/srv/Trigger
```