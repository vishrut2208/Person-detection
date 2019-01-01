# Person-detection
Detecting a person from the crowd with specific detail.

This was final project of my video analytics class where the primary purpose was to identify a person wearing a tshirt with a recognizable logo from the crowd. Second objective was to determine the height of the person given a known dimension object in the frame.

Tasks needed to accomplish:
1.	Work on real-time video (i.e., not captured video)
2.	Video should have few persons (say 2 to 4) moving around and at least one of them should be wearing UTD logo T-shirts. 
3.	You can choose any T-shirt with a logo that is big and easy to recognize. And track the person wearing T-shirt with that Logo.
4.	Your task is to detect and track only the person wearing that logo T-shirt, i.e., put a bounding box on the entire person wearing the T-shirt (not just the T-shirt alone).
5.	Additionally, mark the tracked personâ€™s face and eyes with different colored bounding boxes.
6.	A separate task for the project is to determine the height of the person in feet and inches. To do this you can use an object of known width and height to act as a calibration parameter. You need to track that object and the person, find the ratio and obtain the personâ€™s real height. In order to do this, the person needs to be standing next to the object, at the same depth location.
7.	Person, face, eye, logo and object detection can be done using any OpenCV strategy â€“ no restrictions.

Approach :
> Simple approach was to get the logo segment from the video and compare across the frame, get the coordinates and draw the bounding box.
> Another approach was to train a classifier to identify a specific logo and then search it in the frame to get the coordinates, draw the bounding box
  - Calculating height was easy ratio calculation of the bounding box of known object and the person in a frame and using the ratio to calculate the actual height of the perion
