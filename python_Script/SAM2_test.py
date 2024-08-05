# Import necessary libraries
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio  # Import imageio for GIF creation

# Enable automatic casting for mixed precision training with CUDA
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

# Check if the CUDA device has a major version of 8 or higher and enable TF32 for matrix multiplication
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Import custom function for building the video predictor model
from sam2.build_sam import build_sam2_video_predictor

# Define paths for model checkpoint and configuration
sam2_checkpoint = "checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

# Build the video predictor model
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

# Function to display a mask on the plot
def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        # Generate a random color with transparency
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        # Use a color map for consistent coloring
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

# Function to display points on the plot
def show_points(coords, labels, ax, marker_size=200):
    # Separate positive and negative points based on labels
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    # Plot positive points in green and negative points in red
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


# Directory containing video frames
video_dir = "notebooks/videos/bedroom"


# Get all JPEG frame names in the directory and sort them by frame index
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))



# Display the first frame of the video
frame_idx = 0
fig, ax = plt.subplots(figsize=(12, 8))
plt.title(f"frame {frame_idx}")
ax.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))



# Initialize lists to store points and labels
points = []
labels = []

# Function to handle mouse clicks on the plot
def on_click(event):
    if event.inaxes:
        # Save the clicked point coordinates
        points.append([event.xdata, event.ydata])
        # Ask for the label (1 for positive, 0 for negative)
        label = int(input("Enter label for this point (1 for positive, 0 for negative): "))
        labels.append(label)
        # Show the clicked point on the image with appropriate color
        color = 'green' if label == 1 else 'red'
        ax.scatter(event.xdata, event.ydata, color=color, marker='*', s=200, edgecolor='white', linewidth=1.25)
        plt.draw()


# Connect the click event to the function
cid = fig.canvas.mpl_connect('button_press_event', on_click)

# Show the plot for interaction
plt.show()


# Convert points and labels to numpy arrays for further processing
points = np.array(points, dtype=np.float32)
labels = np.array(labels, dtype=np.int32)

# Initialize the inference state with the predictor model
inference_state = predictor.init_state(video_path=video_dir)

# Use the collected points and labels for processing
ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integer)

# Process the collected points and add new points to the predictor
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# Show the results on the current frame after processing
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

# Run propagation throughout the video and collect the segmentation results in a dictionary
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# Create a list to store images for GIF creation
gif_images = []

# Render the segmentation results every few frames and save them for GIF creation
vis_frame_stride = 1  # Visualize every frame
plt.close("all")
for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {out_frame_idx}")
    frame_image = Image.open(os.path.join(video_dir, frame_names[out_frame_idx]))
    plt.imshow(frame_image)
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
    # Save the current frame to an image buffer for GIF
    plt.axis('off')  # Turn off axes
    plt.tight_layout(pad=0)
    buf = plt.gcf()
    buf.canvas.draw()
    # Use buffer_rgba() and convert RGBA to RGB
    img_data = np.frombuffer(buf.canvas.buffer_rgba(), dtype=np.uint8)
    img_data = img_data.reshape(buf.canvas.get_width_height()[::-1] + (4,))
    img_data_rgb = img_data[:, :, :3]  # Convert RGBA to RGB
    gif_images.append(Image.fromarray(img_data_rgb))
    plt.close()  # Close the figure after processing

# Save the GIF with the predicted results
gif_path = "predicted_results.gif"
gif_images[0].save(
    gif_path,
    save_all=True,
    append_images=gif_images[1:],
    duration=100,  # duration between frames in milliseconds
    loop=0  # loop=0 means the GIF will loop infinitely
)

print(f"GIF saved at {gif_path}")
