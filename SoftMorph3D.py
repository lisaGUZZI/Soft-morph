import torch
import torch.nn as nn
import torch.nn.functional as F



class SoftErosion3D(nn.Module):
    """
    Class implemented using Pytorch module to perform differentiable soft erosion on 3D input image.
    """
    def __init__(self):
        super(SoftErosion3D, self).__init__()
        self.indices_list = torch.tensor([
        [2,0,0], [2,0,1], [2,0,2], [1,0,2], [0,0,2], [0,0,1], [0,0,0], [1,0,0], [1,0,1],
        [2,1,0], [2,1,1], [2,1,2], [1,1,2], [0,1,2], [0,1,1], [0,1,0], [1,1,0],
        [2,2,0], [2,2,1], [2,2,2], [1,2,2], [0,2,2], [0,2,1], [0,2,0], [1,2,0], [1,2,1], [1,1,1]
    ], dtype=torch.long)
        
    def test_format(self, img, connectivity):
        """
        Function to check user inputs :
        - Input image shape must either be [batch_size, channels, depth, height, width] or [depth, height, width].
        - Input image values must be between 0 and 1.
        - Connectivity represents the sutructuring element of the operation. In 3D, it must be either 6, 18 or 26
        """
        dim = img.dim()
        size = img.size()
        if dim > 5 or dim <3:
            raise Exception(f"Invalid input shape {size}. Expected [batch_size, channels, depth, height, width] or [depth, height, width].")
        else:
            if dim == 4 :
                # If the input dimension is 3 it might be due to input format [channels, depth, height, width]
                if size[0] > 3 : # If this is not likely we raise an exception.
                    raise Exception(f"Ambiguous input shape {size}. Expected [batch_size, channels, depth, height, width] or [depth, height, width].")
            for i in range(5-dim):img = img.unsqueeze(0) 
            print("Image resized to : ", img.size())
        if img.min() < 0.0 or img.max() > 1.0:
            raise ValueError("Input image values must be in the range [0, 1].")
        if connectivity not in [6,18,26]:
            raise ValueError("Connectivity should either be 6, 18 or 26")
        return img

    def allcondArithm(self, n, connectivity):
        """
        Apply polynomial formula based on the boolean expression that defines an erosion on each 3x3x3 overlapping cubes of the 3D image.
        Inputs : vector of 3x3x3 overlapping cubes n, connectivity (6,18 or 26) defining the structuring element.
        Output : In binary case returns 0 if the central pixel needs to be changed to 0, returns 1 otherwise.
        """
        if connectivity == 6 :  
            vox = [8, 10, 12, 25, 16, 14, 26]
        elif connectivity == 18 :
            vox = [8, 10, 12, 25, 16, 14, 1,3,5,7,9,11,13,15,18,20,22,24, 26]
        else :
            vox = [8, 10, 12, 25, 16, 14, 1,3,5,7,9,11,13,15,18,20,22,24,0,2,4,6,17,19,21,23,26]
            
        F = torch.prod(n[:, :, :, vox], dim=-1)
        
        return F
    

    def forward(self, im, iterations=1, connectivity = 6):
        """
        Inputs :
        - im : input 3D image of shape [batch_size, channels, depth, height, width] or [depth, height, width].
        - iterations : number of times the morphological operation is repeated.
        - connectivity : connectivity representing the structuring element. Should either be 6, 18 or 26.
        Output : Image after morphological operation
        """
        im = self.test_format(im, connectivity) # Check user inputs
        for _ in range(iterations):
            unfolded = torch.nn.functional.pad(im, (1, 1, 1, 1, 1, 1), mode='constant', value=1)
            unfolded = unfolded.unfold(2, self.cube_size, 1).unfold(3, self.cube_size, 1).unfold(4, self.cube_size, 1)
            unfolded= unfolded.contiguous().view(im.shape[0], im.shape[1], (im.shape[2]*im.shape[3]*im.shape[4]), (self.cube_size**3)) 
            # Apply the formula to all windows simultaneously
            unfolded = unfolded[:, :, :,(self.indices_list[:, 0] * 9) + (self.indices_list[:, 1] * 3) + self.indices_list[0][:, 2]]
            output = self.allcondArithm(unfolded, connectivity)
            # Adjust the dimensions of output to match the spatial dimensions of im
            output = output.view(output.size(0), output.size(1), im.shape[2], im.shape[3], im.shape[4])
            # Element-wise multiplication
            im = im * output
        return im


class SoftDilation3D(nn.Module):
    """
    Class implemented using Pytorch module to perform differentiable soft dilation on 3D input image.
    """
    def __init__(self):
        super(SoftDilation3D, self).__init__()
        self.indices_list = torch.tensor([
        [2,0,0], [2,0,1], [2,0,2], [1,0,2], [0,0,2], [0,0,1], [0,0,0], [1,0,0], [1,0,1],
        [2,1,0], [2,1,1], [2,1,2], [1,1,2], [0,1,2], [0,1,1], [0,1,0], [1,1,0],
        [2,2,0], [2,2,1], [2,2,2], [1,2,2], [0,2,2], [0,2,1], [0,2,0], [1,2,0], [1,2,1], [1,1,1]
    ], dtype=torch.long)

    def test_format(self, img, connectivity):
        """
        Function to check user inputs :
        - Input image shape must either be [batch_size, channels, depth, height, width] or [depth, height, width].
        - Input image values must be between 0 and 1.
        - Connectivity represents the sutructuring element of the operation. In 3D, it must be either 6, 18 or 26
        """
        dim = img.dim()
        size = img.size()
        if dim > 5 or dim <3:
            raise Exception(f"Invalid input shape {size}. Expected [batch_size, channels, depth, height, width] or [depth, height, width].")
        else:
            if dim == 4 :
                # If the input dimension is 3 it might be due to input format [channels, depth, height, width]
                if size[0] > 3 : # If this is not likely we raise an exception.
                    raise Exception(f"Ambiguous input shape {size}. Expected [batch_size, channels, depth, height, width] or [depth, height, width].")
            for i in range(5-dim):img = img.unsqueeze(0) 
            print("Image resized to : ", img.size())
        if img.min() < 0.0 or img.max() > 1.0:
            raise ValueError("Input image values must be in the range [0, 1].")
        if connectivity not in [6,18,26]:
            raise ValueError("Connectivity should either be 6, 18 or 26")
        return img
    
    def allcondArithm(self, n, connectivity):
        """
        Apply polynomial formula based on the boolean expression that defines a dilation on each 3x3x3 overlapping cubes of the 3D image.
        Inputs : vector of 3x3x3 overlapping cubes n, connectivity (6, 18 or 26) defining the structuring element.
        Output : Returns the new value attributed to each central pixel
        """
        if connectivity == 6 :  
            vox = [8, 10, 12, 25, 16, 14, 26]
        elif connectivity == 18 :
            vox = [8, 10, 12, 25, 16, 14, 1,3,5,7,9,11,13,15,18,20,22,24, 26]
        else :
            vox = [8, 10, 12, 25, 16, 14, 1,3,5,7,9,11,13,15,18,20,22,24,0,2,4,6,17,19,21,23, 26]
            
        F = torch.prod(1 - n[:, :, :, vox], dim=-1)
        return 1 - F

    def forward(self, im, iterations=1, connectivity = 6):
        """
        Inputs :
        - im : input 3D image of shape [batch_size, channels, depth, height, width] or [depth, height, width].
        - iterations : number of times the morphological operation is repeated.
        - connectivity : connectivity representing the structuring element. Should either be 6, 18 or 26.
        Output : Image after morphological operation
        """
        im = self.test_format(im, connectivity=6)
        for _ in range(iterations):
            unfolded = torch.nn.functional.pad(im, (1, 1, 1, 1, 1, 1), mode='constant', value=0)
            unfolded = unfolded.unfold(2, self.cube_size, 1).unfold(3, self.cube_size, 1).unfold(4, self.cube_size, 1)
            unfolded= unfolded.contiguous().view(im.shape[0], im.shape[1], (im.shape[2]*im.shape[3]*im.shape[4]), (self.cube_size**3)) 
            # Apply the formula to all windows simultaneously
            unfolded = unfolded[:, :, :,(self.indices_list[:, 0] * 9) + (self.indices_list[:, 1] * 3) + self.indices_list[0][:, 2]]
            output = self.allcondArithm(unfolded, connectivity)
            # Adjust the dimensions of output to match the spatial dimensions of im
            im = output.view(output.size(0), output.size(1), im.shape[2], im.shape[3], im.shape[4])
        return im


class SoftClosing3D(nn.Module):
    """
    Class implemented using Pytorch module to perform differentiable soft closing on 3D input image.
    """
    def __init__(self):
        super(SoftClosing3D, self).__init__()
        self.dilate = SoftDilation3D()
        self.erode = SoftErosion3D()

    def forward(self,input_img, iterations, dilation_connectivity = 6, erosion_connectivity = 6):
        """
        Inputs :
        - im : input 3D image of shape [batch_size, channels, depth, height, width] or [depth, height, width].
        - iterations : number of times each morphological operation is repeated.
        - connectivity : connectivity representing the structuring element. Should either be 6, 18 or 26.
                         Can define different connectivity values for erosion and dilation
        Output : Image after morphological operation
        """
        output = self.dilate(input_img, iterations, dilation_connectivity)
        output = self.erode(output, iterations, erosion_connectivity)
        return output
    
class SoftOpening3D(nn.Module):
    """
    Class implemented using Pytorch module to perform differentiable soft opening on 3D input image.
    """
    def __init__(self):
        super(SoftOpening3D, self).__init__()
        self.erode = SoftErosion3D()
        self.dilate = SoftDilation3D()

    def forward(self,input_img, iterations, dilation_connectivity = 6, erosion_connectivity = 6):
        """
        Inputs :
        - im : input 3D image of shape [batch_size, channels, depth, height, width] or [depth, height, width].
        - iterations : number of times each morphological operation is repeated.
        - connectivity : connectivity representing the structuring element. Should either be 6, 18 or 26.
                         Can define different connectivity values for erosion and dilation
        Output : Image after morphological operation
        """
        output = self.erode(input_img, iterations, erosion_connectivity)
        output = self.dilate(output, iterations, dilation_connectivity)
        return output


class SoftSkeletonizer3D(nn.Module):
    """
    Class implemented using Pytorch module to perform differentiable soft skeletonization on 3D input image.
    
    the max_iter input represents the number of times the thinning operation will be repeated. 
    This input will be automatically determined in future versions.
    """
    def __init__(self, max_iter=5):
        super(SoftSkeletonizer3D, self).__init__()
        self.maxiter = max_iter
        # Extract ordered index list in each subdirection (Up, East, South, Down, West, North)
        self.indices_list = [self.extract_indices(o) for o in range(6)]
        
    def test_format(self, img):
        """
        Function to check user inputs :
        - Input image shape must either be [batch_size, channels, depth, height, width] or [depth, height, width].
        - Input image values must be between 0 and 1.
        """
        dim = img.dim()
        size = img.size()
        if dim > 5 or dim <3:
            raise Exception(f"Invalid input shape {size}. Expected [batch_size, channels, depth, height, width] or [depth, height, width].")
        else:
            if dim == 4 :
                # If the input dimension is 3 it might be due to input format [channels, depth, height, width]
                if size[0] > 3 : # If this is not likely we raise an exception.
                    raise Exception(f"Ambiguous input shape {size}. Expected [batch_size, channels, depth, height, width] or [depth, height, width].")
            for i in range(5-dim):img = img.unsqueeze(0) 
            print("Image resized to : ", img.size())
        if img.min() < 0.0 or img.max() > 1.0:
            raise ValueError("Input image values must be in the range [0, 1].")
        return img
    
    def extract_indices(self, o):
        """
        Function to extract extract ordered index list in each subdirection (North, East, South, West, Up, Down)
        """
        ind = [
        # Up
            torch.tensor([
                [2,0,0], [2,0,1], [2,0,2], [1,0,2], [0,0,2], [0,0,1], [0,0,0], [1,0,0], [1,0,1],
                [2,1,0], [2,1,1], [2,1,2], [1,1,2], [0,1,2], [0,1,1], [0,1,0], [1,1,0],
                [2,2,0], [2,2,1], [2,2,2], [1,2,2], [0,2,2], [0,2,1], [0,2,0], [1,2,0], [1,2,1]
            ], dtype=torch.long),
        # East
            torch.tensor([[2,0,2], [2,1,2], [2,2,2], [1,2,2], [0,2,2], [0,1,2], [0,0,2], [1,0,2], [1,1,2],
                [2,0,1], [2,1,1], [2,2,1], [1,2,1], [0,2,1], [0,1,1], [0,0,1], [1,0,1],
                [2,0,0], [2,1,0], [2,2,0], [1,2,0], [0,2,0], [0,1,0], [0,0,0], [1,0,0], [1,1,0]
                ], dtype=torch.long),
        # South
            torch.tensor([[0,0,0], [0,0,1], [0,0,2], [0,1,2], [0,2,2], [0,2,1], [0,2,0], [0,1,0], [0,1,1],
                [1,0,0], [1,0,1], [1,0,2], [1,1,2], [1,2,2], [1,2,1], [1,2,0], [1,1,0],
                [2,0,0], [2,0,1], [2,0,2], [2,1,2], [2,2,2], [2,2,1], [2,2,0], [2,1,0], [2,1,1]
                ], dtype=torch.long),
        # down
            torch.tensor([[0,2,0], [0,2,1], [0,2,2], [1,2,2], [2,2,2], [2,2,1], [2,2,0], [1,2,0], [1,2,1],
                [0,1,0], [0,1,1], [0,1,2], [1,1,2], [2,1,2], [2,1,1], [2,1,0], [1,1,0],
                [0,0,0], [0,0,1], [0,0,2], [1,0,2], [2,0,2], [2,0,1], [2,0,0], [1,0,0], [1,0,1]
                ], dtype=torch.long),
        # West
            torch.tensor([[2,2,0], [2,1,0], [2,0,0], [1,0,0], [0,0,0], [0,1,0], [0,2,0], [1,2,0], [1,1,0],
                [2,2,1], [2,1,1], [2,0,1], [1,0,1], [0,0,1], [0,1,1], [0,2,1], [1,2,1],
                [2,2,2], [2,1,2], [2,0,2], [1,0,2], [0,0,2], [0,1,2], [0,2,2], [1,2,2], [1,1,2]
                ], dtype=torch.long),
        # North
            torch.tensor([[2,2,0], [2,2,1], [2,2,2], [2,1,2], [2,0,2], [2,0,1], [2,0,0], [2,1,0], [2,1,1],
                [1,2,0], [1,2,1], [1,2,2], [1,1,2], [1,0,2], [1,0,1], [1,0,0], [1,1,0],
                [0,2,0], [0,2,1], [0,2,2], [0,1,2], [0,0,2], [0,0,1], [0,0,0], [0,1,0], [0,1,1]
                ]  , dtype=torch.long) 
        ]
            
        indices = ind[o]
        
        return indices

    def allcondArithm(self, n):
        """
        Apply polynomial formula based on the boolean expression that defines a thinning operation on each 3x3x3 overlapping cubes of the 3D image.
        Inputs : vector of 3x3x3 overlapping cubes n.
        Output : In binary case returns 0 if the central pixel needs to be changed to 0, returns 1 otherwise.
        """
        M1 = (1-n[:, :, :, 0])*(1-n[:, :, :, 1])*(1-n[:, :, :, 2])*(1-n[:, :, :, 3])*(1-n[:, :, :, 4])*(1-n[:, :, :, 5])*(1-n[:, :, :, 6]) *(1-n[:, :, :, 7])*(1-n[:, :, :, 8])*n[:, :, :, 25]*(1-((1-n[:, :, :, 9])*(1-n[:, :, :, 10])*(1-n[:, :, :, 11])*(1-n[:, :, :, 12])*(1-n[:, :, :, 13])*(1-n[:, :, :, 14])*(1-n[:, :, :, 15])*(1-n[:, :, :, 16])*(1-n[:, :, :, 17])*(1-n[:, :, :, 18])*(1-n[:, :, :, 19])*(1-n[:, :, :, 20])*(1-n[:, :, :, 21])*(1-n[:, :, :, 22])*(1-n[:, :, :, 23])*(1-n[:, :, :, 24])))
        M2 = ((1-n[:, :, :, 8])*n[:, :, :, 25]) * (1-((1-((1-n[:, :, :, 3])*(1-n[:, :, :, 4])*(1-n[:, :, :, 5])*(1-n[:, :, :, 6])*(1-n[:, :, :, 7])*n[:, :, :, 10]))*(1-((1-n[:, :, :, 1])*(1-n[:, :, :, 2])*(1-n[:, :, :, 3])*(1-n[:, :, :, 4])*(1-n[:, :, :, 5])*n[:, :, :, 16]))*(1-((1-n[:, :, :, 0])*(1-n[:, :, :, 1])*(1-n[:, :, :, 2])*(1-n[:, :, :, 3])*(1-n[:, :, :, 7])*n[:, :, :, 14]))*(1-((1-n[:, :, :, 0])*(1-n[:, :, :, 1])*(1-n[:, :, :, 5])*(1-n[:, :, :, 6])*(1-n[:, :, :, 7])*n[:, :, :, 12]))))
        M3 = ((1-n[:, :, :, 8])*n[:, :, :, 25]) * (1 - ((1-((1-n[:, :, :, 5])*(1-n[:, :, :, 6])*(1-n[:, :, :, 7])*n[:, :, :, 10]*n[:, :, :, 12]))*(1-((1-n[:, :, :, 5])*(1-n[:, :, :, 4])*(1-n[:, :, :, 3])*n[:, :, :, 10]*n[:, :, :, 16]))*(1-((1-n[:, :, :, 1])*(1-n[:, :, :, 2])*(1-n[:, :, :, 3])*n[:, :, :, 16]*n[:, :, :, 14]))*(1-((1-n[:, :, :, 0])*(1-n[:, :, :, 1])*(1-n[:, :, :, 7])*n[:, :, :, 14]*n[:, :, :, 12]))))
        M4 = ((1-n[:, :, :, 1])*(1-n[:, :, :, 3])*(1-n[:, :, :, 5])*(1-n[:, :, :, 7])*(1-n[:, :, :, 8])*n[:, :, :, 25]) * (1-((1-((1-n[:, :, :, 0])*(1-n[:, :, :, 4])*(1-n[:, :, :, 6])*n[:, :, :, 11]*n[:, :, :, 2]))*(1-((1-n[:, :, :, 2])*(1-n[:, :, :, 4])*(1-n[:, :, :, 6])*n[:, :, :, 0]*n[:, :, :, 9]))*(1-((1-n[:, :, :, 0])*(1-n[:, :, :, 4])*(1-n[:, :, :, 2])*n[:, :, :, 6]*n[:, :, :, 15]))*(1-((1-n[:, :, :, 0])*(1-n[:, :, :, 6])*(1-n[:, :, :, 2])*n[:, :, :, 4]*n[:, :, :, 13]))))
        M5 = ((1-n[:, :, :, 0])*(1-n[:, :, :, 1])*(1-n[:, :, :, 2])*(1-n[:, :, :, 3])*(1-n[:, :, :, 4])*(1-n[:, :, :, 5])*(1-n[:, :, :, 6])*(1-n[:, :, :, 7])*(1-n[:, :, :, 8])*(1-n[:, :, :, 25])) * (1 - ((1-((1-n[:, :, :, 13])*(1-n[:, :, :, 14])*(1-n[:, :, :, 15])*(1-n[:, :, :, 21])*(1-n[:, :, :, 22])*(1-n[:, :, :, 23])*n[:, :, :, 18]*(1-((1-n[:, :, :, 9])*(1-n[:, :, :, 10])*(1-n[:, :, :, 11])*(1-n[:, :, :, 12])*(1-n[:, :, :, 16])*(1-n[:, :, :, 17])*(1-n[:, :, :, 19])*(1-n[:, :, :, 20])*(1-n[:, :, :, 24])))))*(1-((1-n[:, :, :, 13])*(1-n[:, :, :, 12])*(1-n[:, :, :, 11])*(1-n[:, :, :, 21])*(1-n[:, :, :, 20])*(1-n[:, :, :, 19])*n[:, :, :, 24]*(1-((1-n[:, :, :, 9])*(1-n[:, :, :, 10])*(1-n[:, :, :, 14])*(1-n[:, :, :, 15])*(1-n[:, :, :, 16])*(1-n[:, :, :, 17])*(1-n[:, :, :, 18])*(1-n[:, :, :, 22])*(1-n[:, :, :, 23])))))*(1-((1-n[:, :, :, 9])*(1-n[:, :, :, 10])*(1-n[:, :, :, 11])*(1-n[:, :, :, 17])*(1-n[:, :, :, 18])*(1-n[:, :, :, 19])*n[:, :, :, 22]*(1-((1-n[:, :, :, 15])*(1-n[:, :, :, 14])*(1-n[:, :, :, 13])*(1-n[:, :, :, 12])*(1-n[:, :, :, 16])*(1-n[:, :, :, 23])*(1-n[:, :, :, 21])*(1-n[:, :, :, 20])*(1-n[:, :, :, 24])))))*(1-((1-n[:, :, :, 9])*(1-n[:, :, :, 16])*(1-n[:, :, :, 15])*(1-n[:, :, :, 24])*(1-n[:, :, :, 17])*(1-n[:, :, :, 23])*n[:, :, :, 20]*(1-((1-n[:, :, :, 14])*(1-n[:, :, :, 10])*(1-n[:, :, :, 11])*(1-n[:, :, :, 12])*(1-n[:, :, :, 13])*(1-n[:, :, :, 18])*(1-n[:, :, :, 19])*(1-n[:, :, :, 22])*(1-n[:, :, :, 21])))))))
        M6 = ((1-n[:, :, :, 0])*(1-n[:, :, :, 1])*(1-n[:, :, :, 2])*(1-n[:, :, :, 3])*(1-n[:, :, :, 4])*(1-n[:, :, :, 5])*(1-n[:, :, :, 6])*(1-n[:, :, :, 7])*(1-n[:, :, :, 8])*(1-n[:, :, :, 25])) * (1 - ((1-((1-n[:, :, :, 14])*(1-n[:, :, :, 15])*(1-n[:, :, :, 16])*(1-n[:, :, :, 22])*(1-n[:, :, :, 23])*(1-n[:, :, :, 24])*n[:, :, :, 18]*n[:, :, :, 20]))*(1-((1-n[:, :, :, 14])*(1-n[:, :, :, 12])*(1-n[:, :, :, 13])*(1-n[:, :, :, 22])*(1-n[:, :, :, 20])*(1-n[:, :, :, 21])*n[:, :, :, 18]*n[:, :, :, 24]))*(1-((1-n[:, :, :, 10])*(1-n[:, :, :, 11])*(1-n[:, :, :, 12])*(1-n[:, :, :, 18])*(1-n[:, :, :, 19])*(1-n[:, :, :, 20])*n[:, :, :, 24]*n[:, :, :, 22]))*(1-((1-n[:, :, :, 9])*(1-n[:, :, :, 10])*(1-n[:, :, :, 16])*(1-n[:, :, :, 24])*(1-n[:, :, :, 17])*(1-n[:, :, :, 18])*n[:, :, :, 22]*n[:, :, :, 20]))))

        F = 1-((1-M1)*(1-M2)*(1-M3)*(1-M4)*(1-M5)*(1-M6))
        F = 1-F
        return F  

    def forward(self, im):
        """
        Input :
        - im : input 3D image of shape [batch_size, channels, depth, height, width] or [depth, height, width].
        Output : Image after morphological operation
        """
        im = self.test_format(im)
        for _ in range(self.maxiter):
            for o in range(6):  # Iterate over all 6 orientations
                unfolded = torch.nn.functional.pad(im, (1, 1, 1, 1, 1, 1), mode='constant', value=0)
                # Reshape to get every 3x3x3 overlapping squares with a stride of 1 
                unfolded = unfolded.unfold(2, self.cube_size, 1).unfold(3, self.cube_size, 1).unfold(4, self.cube_size, 1)
                unfolded= unfolded.contiguous().view(im.shape[0], im.shape[1], (im.shape[2]*im.shape[3]*im.shape[4]), (self.cube_size**3)) 
                # Apply the formula to all cubes simultaneously
                unfolded = unfolded[:, :, :,(self.indices_list[o][:, 0] * 9) + (self.indices_list[o][:, 1] * 3) + self.indices_list[o][:, 2]]
                output = self.allcondArithm(unfolded)
                # # Adjust the dimensions of output to match the spatial dimensions of im
                output = output.view(output.size(0), output.size(1), im.shape[2], im.shape[3], im.shape[4])
                # Element-wise multiplication
                im = im*output
        return im   
