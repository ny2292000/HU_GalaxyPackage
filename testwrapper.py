from HU_Galaxy import GalaxyWrapper


# Create a new GalaxyWrapper instance with some parameters
gw = GalaxyWrapper(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1, 1, 1, 1, 1)

# Call the get_f_z function with some parameters
result = gw.get_f_z([1.0, 2.0, 3.0])

# Print the result
print(result)
