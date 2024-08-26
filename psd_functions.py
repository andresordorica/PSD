import numpy as np
import matplotlib.pyplot as plt
from random import uniform
import json
import os 
from tqdm import tqdm
import cupy as cp  

'''
                Mantina M, Chamberlin AC, Valero R, Cramer CJ, Truhlar DG. 
                Consistent van der Waals radii for the whole main group.
                J Phys Chem A. 2009 May 14;113(19):5806-12.
                doi: 10.1021/jp8111556. PMID: 19382751; PMCID: PMC3658832.
                '''
bondi_radii =  { 
                
                
                "C" : 0.17, 
                "N": 0.155,
                "O": 0.152,
                "F": 0.147,
                "Si": 0.210,
                "P": 0.180,
                "S": 0.180, 
                "Cl": 0.175 ,
                "Br":0.183 ,
                "H": 0.110,
                
}



def transcribe_gro_radius(gro_file_path):
    import re
    # Open the .gro file
    with open(gro_file_path, "r") as gro_file:
        # Read all lines
        lines = gro_file.readlines()

    # Initialize lists to store coordinates
    x_coords = []
    y_coords = []
    z_coords = []
    names = []
    index = []
    full_name = []
    i = 0
    # Skip the first two lines and the last line
    for line in lines[1:]:
        # Split the line into columns
        columns = line.split()
        if len(columns) >= 6:
            #print(columns)
            #print(len(columns))
            # Extract coordinates and convert them to floats
            names.append(re.sub(r'\d+', '', columns[1]))  # Strip digits from the string
            full_name.append(str(columns[1]))
            index.append(str(i))
            x_coords.append(float(columns[3]))
            y_coords.append(float(columns[4]))
            z_coords.append(float(columns[5]))
            i+=1
        elif len(columns) == 5 and columns[0]!='wetting' :
            #print(columns)
            #print(len(columns))
            names.append(re.sub(r'\d+', '', columns[1]))  # Strip digits from the string
            full_name.append(str(columns[1]))
            index.append(str(i))
            x_coords.append(float(columns[2]))
            y_coords.append(float(columns[3]))
            z_coords.append(float(columns[4]))
            i+=1
            
        
            
            
      

    print("Length of Names:", len(names))
    print("Length of Indexes:", len(index))
    print("Length of X Coordinates:", len(x_coords))
    print("Length of Y Coordinates:", len(y_coords))
    print("Length of Z Coordinates:", len(z_coords))
    radius_list = [bondi_radii[atom_name] for atom_name in names]
    print("Length of Radius list:", len(radius_list))
    print("Radius list:",radius_list)   
    
    max_x = max(x_coords)
    max_y = max(y_coords)
    max_z = max(z_coords)
    
    min_x = min(x_coords)
    min_y = min(y_coords)
    min_z = min(z_coords)
    
    print(max_x, min_x)
    print(max_y, min_y)
    print(max_z, min_z)

    # Create a list of dictionaries
    data = []
    for i, element in enumerate(full_name):
        dictionary = {'name': full_name[i], 'x': x_coords[i], 'y': y_coords[i], 'z': z_coords[i], 'radius':radius_list[i]}
        data.append(dictionary)
    box_d = np.array([max_x,max_y,max_z])
    print(f"Box dimensions (nm):{box_d}")
    return data, box_d

def PSD(data, number_grid_points = 1, box_dimensions_array = [6,6,6], step_size = 0.010, units = "nm", diameter=False, Gaussian_Fit = True, GPU =False):
    
    if GPU:
        np_to_use = cp
    else:
        np_to_use = np
    
    print("Box dimensions (nm)")
    print(box_dimensions_array)
    # Convert dots_data to NumPy array
    dots_data = np_to_use.array([(dot["x"], dot["y"], dot["z"], dot["radius"]) for dot in data])
    # Define the dimensions of the initial box
    box_dimensions = {'x': box_dimensions_array[0], 'y': box_dimensions_array[1], 'z': box_dimensions_array[2]}  # Example dimensions
    # Number of cubes along each axis
    n_cubes = int(number_grid_points * box_dimensions['x'])
    # Calculate the cube length along each axis
    cube_length_x = box_dimensions['x'] / n_cubes
    cube_length_y = box_dimensions['y'] / n_cubes
    cube_length_z = box_dimensions['z'] / n_cubes
    # Generate grid cube centers
    grid_x, grid_y, grid_z = np_to_use.meshgrid(np_to_use.arange(0, box_dimensions['x'], cube_length_x),
                                        np_to_use.arange(0, box_dimensions['y'], cube_length_y),
                                        np_to_use.arange(0, box_dimensions['z'], cube_length_z),
                                        indexing='ij')

    grid_centers = np_to_use.column_stack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel()))
    # Initialize the largest non-overlapping sphere radius
    grid_cube_volume = cube_length_x * cube_length_y * cube_length_z
    total_calculations = len(grid_centers) * len(data)

    print(f"Grid cube volume (nm^3): {(grid_cube_volume)}")
    print(f"Grid cube size (nm): x:{(cube_length_x)}, y:{cube_length_y}, z:{cube_length_z}")
    print(f"Number of computations: {(total_calculations)}")
    print(f"Number of iterations: {len(grid_centers)}")
    # Iterate over each grid cube center
    # Function to check if a sphere overlaps with existing dots (vectorized)
    def sphere_overlaps(candidate_centers, candidate_radii, dots_data):
        dot_centers = dots_data[:, :3]
        dot_radii = dots_data[:, 3]
        distances = np_to_use.linalg.norm(candidate_centers[:, None] - dot_centers, axis=2)
        overlaps = distances < (candidate_radii[:, None] + dot_radii)
        return np_to_use.any(overlaps, axis=1)
    
    def check_overlap_optimized(new_point, new_radius, sphere_list):
        centers = np.array([center for center, _ in sphere_list])
        radii = np.array([radius for _, radius in sphere_list])
        
        '''
        #DebugBlock
        print("Length sphere list:")
        print(len(sphere_list))
        print("New point, new radius", flush = True)
        print(new_point)
        print("new radius", flush = True)
        print(new_radius)
        
        # Debugging: Print shapes
        print("Shape of centers:", centers.shape)  # Expected (N, 3)
        print("Shape of new_point:", new_point.shape)  # Expected (3,)
        print("Shape of radii:", radii.shape)  # Expected (N,)
        '''
        # Calculate squared distances for speed (avoiding sqrt operation)
        squared_distances = np.sum((centers - new_point) ** 2, axis=1)
        squared_radii_sum = (radii + new_radius) ** 2
        
        # Find indices of all overlapping spheres
        overlap_indices = np.where(squared_distances < squared_radii_sum)[0]
        
        return overlap_indices if len(overlap_indices) > 0 else None


    # Initialize a list to store the radii of inserted spheres
    inserted_spheres_radii = []
    radii_list_total = [0.0000, ] # was 0.0000
    steo_ = step_size
    sphere_list = []
    # Iterate over each grid cube center
    for center in tqdm(grid_centers, desc="Processing grid centers"):
        # Initialize the radius of the inserted sphere
        radius_inserted_sphere = 0.0001 # Value from pore blazer
        # Keep increasing the radius of the inserted sphere until it overlaps with any existing dots
        ci = 0
        while True:
            # Check if the candidate sphere overlaps with any existing dots
            overlaps = sphere_overlaps(np_to_use.array([center]), np_to_use.array([radius_inserted_sphere]), dots_data)
            if overlaps.any():
                break  # Exit the loop if overlap is detected
            
            # Increment the radius of the inserted sphere
            radii_list_total.append(radius_inserted_sphere)
            radius_inserted_sphere += steo_
            ci +=1
        
        # Add the radius of the inserted sphere to the list
        if ci > 5 :
            new_point = np_to_use.array(center)
            
            result_tuple = (new_point, radius_inserted_sphere)
            
            if len(sphere_list) > 0:
                overlap_indices = check_overlap_optimized(new_point, radius_inserted_sphere, sphere_list)
                
                if overlap_indices is not None:
                    # Overlap detected; iterate over all overlapping spheres
                    replace = True
                    for index in overlap_indices:
                        existing_radius = sphere_list[index][1]
                        if radius_inserted_sphere <= existing_radius:
                            replace = False
                            print(f"Kept the existing sphere at index {index} with radius {existing_radius}.", flush = True)
                        else:
                            # Replace the smaller overlapping sphere
                            sphere_list[index] = (new_point, radius_inserted_sphere)
                            print(f"Replaced existing sphere at index {index} with the new one.", flush = True)
                    
                    if replace:
                        # If we replaced smaller spheres, remove them from the list
                        sphere_list = [sphere_list[i] for i in range(len(sphere_list)) if i not in overlap_indices or i == overlap_indices[0]]
                else:
                    # No overlap, add the new sphere
                    sphere_list.append((new_point, radius_inserted_sphere))
                    print("Added the new sphere to the list.", flush = True)
            else:
                # Sphere list is empty, add the new sphere
                sphere_list.append((new_point, radius_inserted_sphere))
                print("Added the new sphere to the empty list.", flush = True)
                
                
         
              
           
    print(sphere_list)
    # Plot the histogram of the radii of inserted spheres
    inserted_spheres_radii = [radius for _, radius in sphere_list]
    unique_radii, counts = np_to_use.unique(inserted_spheres_radii, return_counts=True)

    
    other_radii = np_to_use.arange(0, unique_radii[0] , steo_)
    unique_radii_2 = np_to_use.concatenate((other_radii[:-1], unique_radii))
    zero_array = np_to_use.zeros_like(other_radii[:-1])
    counts_2 = np_to_use.concatenate((zero_array, counts))
    #################################
    unique_radii = unique_radii_2
    counts = counts_2
    counts1 = counts[::-1]
    # Compute the cumulative distribution function (CDF)
    cdf = np_to_use.cumsum(counts1) / sum(counts1)
    cdf_reversed =  cdf[::-1]
    derivative = -1*np_to_use.gradient(cdf_reversed, unique_radii)
    
    if diameter == True:
        unique_radii = unique_radii*2
    
    if units =='A':
        unique_radii = unique_radii *10
        
    ##################################Guassian Fit###########################
    if Gaussian_Fit ==True:
        from scipy.optimize import curve_fit
        # Define Gaussian function
        x_data = unique_radii
        y_data = derivative
        def gaussian(x, amplitude, mean, stddev):
            return amplitude * np.exp(-0.5 * ((x - mean) / stddev) ** 2)

        # Initial guess for parameters
        initial_guess = [1, 0, 1]

        # Fit Gaussian to data
        params, covariance = curve_fit(gaussian, x_data, y_data, p0=initial_guess)
        # Generate more points for smoother plot
        x_fit = np.linspace(-5, 5, 1000)
        y_fit = gaussian(x_fit, *params)
        y_fit = y_fit/np.max(y_fit)
    else:
        x_fit = []
        y_fit = []
        params = []
    ########################################################################
    
    return unique_radii, counts, cdf_reversed,  derivative, x_fit, y_fit, params, sphere_list

def plot_PSD(path, unique_radii_L,counts,cdf_reversed, derivative, units = "A", Diameter = True, save = True, case = "NONE"  ):
    
    
    unique_radii = np.array(unique_radii_L)
    unique_radii = unique_radii.astype(float)
    if Diameter == True:
        x_label = "Diameter {}".format(units)
    else:
        x_label = "Radius {}".format(units)
        
    plt.figure(figsize=(10, 6))  # Width: 10 inches, Height: 6 inches
    plt.scatter(unique_radii, counts, edgecolor='black')
    plt.xlabel(x_label)
    plt.ylabel('Frequency')
    plt.title(f' {case} Histogram of Inserted Sphere Radii')
    if save == True:
        plt.savefig(os.path.join(path, "PSD_histogram.png"))
        
    plt.show()

    # Plot the CDF
    plt.figure(figsize=(10, 6))  # Width: 10 inches, Height: 6 inches
    plt.plot(unique_radii, cdf_reversed, marker='o', linestyle='-')
    plt.xlabel(x_label)
    plt.ylabel('Cumulative Pore Size Distribution')
    plt.title(f' {case} Cumulative Distribution ')
    plt.grid(True)
    if save == True:
        plt.savefig(os.path.join(path, "PSD_Cumulative.png"))
    plt.show()

    # Plot the derivative of -dV/dr 
   
    max_index = np.argmax(derivative)
    max_value = derivative[max_index]
    max_x = unique_radii[max_index]
    max_string = round(unique_radii[max_index], 3)

    ############3
    plt.figure(figsize=(10, 6))  # Width: 10 inches, Height: 6 inches
    plt.plot(unique_radii, derivative, marker='o', linestyle='-')
    plt.axvline(x=max_x, color='r', linestyle='-.')
    plt.xlabel(x_label)
    plt.ylabel('PSD')
    plt.title(f' {case} Pore size distribution vs radii ')
    plt.grid(True)
    plt.text(unique_radii[-5], (max_value - (0.20*max_value)), f'Dp: {max_string} A', fontsize=12, ha='right')
    if save == True:
        plt.savefig(os.path.join(path, "PSD_Derivative.png"))
    plt.show()

    '''
    ###################Gaussian FIt
    # Plot original data
    plt.scatter(unique_radii, derivative, label='Data')
    # Plot fitted Gaussian with more points
    plt.plot(x_fit, y_fit, color='red', label='Gaussian Fit')
    # Add labels and legend
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.ylim(0, max(y_fit))
    # Add annotations for mean and standard deviation
    mean = params[1]
    stddev = params[2]
    plt.text(mean, 0.1, f'Mean: {mean:.2f}', color='blue', ha='center')
    plt.text(mean, 0.5, f'Std Dev: {stddev:.2f}', color='blue', ha='center')
    # Show plot
    plt.show()
    '''


