import pygame
import numpy as np
import random
import os
import sys

if __name__ == "__main__":
    """
    Program to run Visual Turing Test, a method of evaluation of the model using human participants. See the paper for
    details on the idea behind the VTT and why it is a necessary metric.
    
    NOTE: The user must run test_refinement.py first to obtain the image files to VTT will use!
    
    Usage:
    python turing_test.py <# of images to see>
    """

    # Number of image pairs observer will see
    n_trials = int(sys.argv[-1])

    # Initialize Pygame
    pygame.init()

    # Set up the screen
    screen_width = 2 * 224
    screen_height = 224
    screen = pygame.display.set_mode((screen_width, screen_height), pygame.DOUBLEBUF)
    pygame.display.set_caption("Visual Turing Test")

    # Set up the clock
    clock = pygame.time.Clock()

    real_images_folder = "test/refinement/ground_truth"
    fake_images_folder = "test/refinement/colorized_1_5"

    n_real_images = len(os.listdir(real_images_folder))

    indices = np.arange(n_real_images)
    indices = np.random.choice(indices, size=n_trials, replace=False)

    # Main loop
    running = True

    # Initialize counters
    trial_count = 0
    n_correct = 0
    n_incorrect = 0

    while running:
        # Check for events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        idx = indices[trial_count]

        # Load fake and real images
        real_image = pygame.image.load(real_images_folder + "/" + str(idx) + ".png")
        fake_image = pygame.image.load(fake_images_folder + "/" + str(idx) + ".png")

        # Display the images side by side, fake and real images will be in a random order
        screen.fill((255, 255, 255))
        left = bool(random.getrandbits(1))
        if left:
            screen.blit(real_image, (0, 0))
            screen.blit(fake_image, (224, 0))
        else:
            screen.blit(real_image, (224, 0))
            screen.blit(fake_image, (0, 0))

        # Update the display
        pygame.display.flip()

        # Wait for the user to click on an image
        clicked = False
        correct = False
        while not clicked:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    clicked = True

                    # Check which image was clicked
                    mouse_x, mouse_y = pygame.mouse.get_pos()

                    if mouse_x < screen_width // 2:
                        if left:
                            correct = True
                    else:
                        if not left:
                            correct = True

        # Update counters
        if correct:
            n_correct += 1
        else:
            n_incorrect += 1

        # Increment trial count
        trial_count += 1

        # Check if we've reached the end of the test
        if trial_count >= n_trials:
            running = False

    # Quit Pygame
    pygame.quit()

    # Print results of VTT
    print("Turing Test Complete!")
    print("Seen:", trial_count)
    print("Correct:", n_correct)
    print("Incorrect:", n_incorrect)
    print("Incorrect %:", n_incorrect / trial_count * 100)
