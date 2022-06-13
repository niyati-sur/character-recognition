# character-recognition

## help 
### git commands
Clone a copy using 
> git clone https://github.com/niyati-sur/character-recognition

See difference using 
> git diff

Add your changes 
> git add <file_name>

commit the changes
> git commit -m "my changes"
> git push

### image operations
#### read images
```{python}
hsf_image = cv2.imread(r'D:\by_class\by_class\4d\hsf_0\hsf_0_00000.png')
```
#### check image size
```{python}
hsf_image.shape
```
#### convert a rgb to grey
```
hsf_image_gray = cv2.cvtColor(hsf_image, cv2.COLOR_BGR2GRAY)
```
#### resize an image
```
hsf_image_resize = cv2.resize(hsf_image_gray, (28, 28))
```
#### flatten an image
```
hsf_image_resize.flatten()
```
