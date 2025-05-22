import cv2
import numpy as np
import torch
import os
from mtcnn import MTCNN
from facenet_pytorch import InceptionResnetV1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Initialize face detection and recognition models
detector = MTCNN()
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def detect_faces(image):
    """
    Detect faces in an image using MTCNN
    
    Args:
        image: BGR image from OpenCV
        
    Returns:
        boxes: list of bounding boxes in format [x1, y1, x2, y2]
        probs: list of detection probabilities
    """
    # Convert BGR to RGB (OpenCV uses BGR by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    result = detector.detect_faces(image_rgb)
    
    boxes = []
    probs = []
    
    for face in result:
        box = face['box']
        # Convert from [x, y, width, height] to [x1, y1, x2, y2]
        box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
        confidence = face['confidence']
        
        boxes.append(box)
        probs.append(confidence)
    
    return boxes, probs

def draw_faces(image, boxes, names=None, similarities=None, recognition_times=None):
    """
    Draw bounding boxes and labels on the image
    
    Args:
        image: BGR image from OpenCV
        boxes: list of bounding boxes in format [x1, y1, x2, y2]
        names: list of names for each face (optional)
        similarities: list of similarity scores (optional)
        recognition_times: list of recognition times (optional)
        
    Returns:
        image: image with bounding boxes and labels drawn
    """
    image_copy = image.copy()
    
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        # Determine color based on recognition status
        if names is None or names[i] == "Unknown":
            color = (0, 0, 255)  # Red for unknown
        elif recognition_times is not None and recognition_times[i] >= 5:
            color = (0, 255, 0)  # Green for logged attendance
        else:
       
            color = (255, 0, 0)  # Blue for recognized but not logged
        
        # Draw bounding box
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)
        
        # Draw name and similarity if available
        if names is not None:
            label = names[i]
            if similarities is not None:
                label += f": {similarities[i]:.2f}"
            cv2.putText(image_copy, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw recognition time if available
        if recognition_times is not None and names is not None and names[i] != "Unknown":
            cv2.putText(image_copy, f"Time: {recognition_times[i]:.1f}s", (x1, y2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    return image_copy

def get_face_embedding(image, box):
    """
    Extract face embedding using pytorch-facenet
    
    Args:
        image: BGR image from OpenCV
        box: bounding box in format [x1, y1, x2, y2]
        
    Returns:
        embedding: face embedding vector
    """
    # Extract face from image
    x1, y1, x2, y2 = box
    face = image[y1:y2, x1:x2]
    
    # Resize to required input size for facenet
    face = cv2.resize(face, (160, 160))
    
    # Convert to RGB if needed
    if len(face.shape) == 2:
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
    elif face.shape[2] == 4:
        face = face[:, :, :3]
    
    # Convert to RGB if in BGR format
    if face.shape[2] == 3:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    
    # Normalize pixel values
    face = face.transpose((2, 0, 1))
    face = torch.from_numpy(face).float()
    face = face.unsqueeze(0) / 255.0
    
    # Get embedding
    with torch.no_grad():
        embedding = facenet(face)
    
    return embedding.cpu().numpy()[0]

def load_registered_users(embeddings_dir="data/embeddings"):
    """
    Load all registered user embeddings
    
    Args:
        embeddings_dir: directory containing user embeddings
        
    Returns:
        users: dictionary of user names and their embeddings
    """
    users = {}
    
    if not os.path.exists(embeddings_dir):
        os.makedirs(embeddings_dir)
        return users
    
    for file in os.listdir(embeddings_dir):
        if file.endswith(".npy"):
            name = file[:-4]  # Remove .npy extension
            embeddings = np.load(f"{embeddings_dir}/{file}")
            users[name] = embeddings
    
    return users

def recognize_face(embedding, registered_users, threshold=0.7):
    """
    Recognize a face by comparing its embedding with registered users
    
    Args:
        embedding: face embedding vector
        registered_users: dictionary of registered users and their embeddings
        threshold: similarity threshold
        
    Returns:
        name: recognized user name or "Unknown"
        similarity: highest similarity score
    """
    if not registered_users:
        return "Unknown", 0
    
    max_similarity = 0
    recognized_name = "Unknown"
    
    for name, user_embeddings in registered_users.items():
        for user_embedding in user_embeddings:
            # Compute cosine similarity
            similarity = np.dot(embedding, user_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(user_embedding))
            
            if similarity > max_similarity:
                max_similarity = similarity
                recognized_name = name
    
    if max_similarity < threshold:
        return "Unknown", max_similarity
    
    return recognized_name, max_similarity

def detect_and_recognize_faces(image, registered_users):
    """
    Detect and recognize faces in an image
    
    Args:
        image: BGR image from OpenCV
        registered_users: dictionary of registered users and their embeddings
        
    Returns:
        boxes: list of bounding boxes
        names: list of recognized names
        similarities: list of similarity scores
    """
    # Detect faces
    boxes, _ = detect_faces(image)
    
    names = []
    similarities = []
    
    # Recognize each face
    for box in boxes:
        # Get face embedding
        embedding = get_face_embedding(image, box)
        
        # Recognize face
        name, similarity = recognize_face(embedding, registered_users)
        
        names.append(name)
        similarities.append(similarity)
    
    return boxes, names, similarities

def process_registration_frame(frame, current_embeddings):
    """
    Process a frame for registration
    
    Args:
        frame: BGR image from OpenCV
        current_embeddings: list of current embeddings
        
    Returns:
        processed_frame: frame with faces drawn
        can_capture: True if a face can be captured
        face_box: detected face box if can_capture is True
    """
    # Detect faces
    boxes, probs = detect_faces(frame)
    
    # Draw faces on frame
    processed_frame = draw_faces(frame, boxes)
    
    # Add registration instructions
    cv2.putText(processed_frame, f"Captured {len(current_embeddings)}/5 embeddings", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Check if exactly one face is detected
    can_capture = len(boxes) == 1
    face_box = boxes[0] if can_capture else None
    
    return processed_frame, can_capture, face_box

def save_user_embedding(name, embeddings, embeddings_dir="data/embeddings"):
    """
    Save user embeddings to file
    
    Args:
        name: name of the user
        embeddings: list of face embeddings
        embeddings_dir: directory to save embeddings
        
    Returns:
        success: True if saved successfully
    """
    if not os.path.exists(embeddings_dir):
        os.makedirs(embeddings_dir)
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings)
    
    # Save to file
    np.save(f"{embeddings_dir}/{name}.npy", embeddings_array)
    
    return True