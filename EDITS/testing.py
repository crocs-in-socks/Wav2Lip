from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
import soundfile as sf
import cv2
import subprocess
from tqdm import tqdm

def clip_video(input_file, output_file, start_time, end_time):
    ffmpeg_extract_subclip(input_file, start_time, end_time, targetname=output_file)
    return output_file

def clip_audio(input_file, output_file, start_time, end_time):
    audio_data, sample_rate = sf.read(input_file)
    start_frame = int(start_time * sample_rate)
    end_frame = int(end_time * sample_rate)

    end_frame = min(end_frame, len(audio_data))

    audio_segment = audio_data[start_frame:end_frame]
    sf.write(output_file, audio_segment, sample_rate)
    return output_file

def get_video_length(input_file):
    video = VideoFileClip(input_file)
    duration = video.duration
    video.close()
    return duration

def get_audio_length(input_file):
    audio_data, sample_rate = sf.read(input_file)
    duration = len(audio_data) / sample_rate
    return duration

def match_lengths(input_video, output_video, input_audio, output_audio):
    if get_audio_length(input_audio) < get_video_length(input_video):
        print('The video clip is longer than the given audio clip.')
        clip_video(input_video, output_video, 0, get_audio_length(input_audio))

def replace_audio(input_video, input_audio, output_video):
    video = VideoFileClip(input_video)
    audio = AudioFileClip(input_audio)

    dubbed_video = video.set_audio(audio)
    dubbed_video.write_videofile(output_video, codec='libx264', audio_codec='aac')

def run_inference(checkpoint_path, video_path, audio_path, output_path):
    command = [
        'python', 'inference.py',
        '--checkpoint_path', checkpoint_path,
        '--face', video_path,
        '--audio', audio_path,
        '--outfile', output_path,
    ]
    subprocess.run(command)

def get_timestamps(input_video, threshold=30):

    print('Preprocessing.')

    video_capture = cv2.VideoCapture(input_video)
    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    no_face_timestamps = []
    face_timestamps = []
    current_frame = 0
    start_time = None
    prev_frame = None
    mean_diff = 0

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    for i in tqdm(range(total_frames)):
        ret, frame = video_capture.read()
        if not ret:
            break

        if prev_frame is not None:
            diff = cv2.absdiff(frame, prev_frame)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            mean_diff = gray_diff.mean()

        if mean_diff > threshold and start_time is None:
            # Image changes
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            # print()
            # print("start", mean_diff, i / frame_rate)
            # print()
            # if len(faces) != 1:
            start_time = i / frame_rate

        elif mean_diff > threshold and start_time is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            # print()
            # print("end before check", mean_diff,  i / frame_rate)
            # print()
            if len(faces) == 1:
                end_time = i / frame_rate
                no_face_timestamps.append(start_time)
                no_face_timestamps.append(end_time)
                start_time = None
                # print()
                # print("end after check", mean_diff, end_time)
                # print()
                # cv2.imwrite(f'static{i}.jpg', frame)

        prev_frame = frame
    
    if start_time is not None:
        end_time = i / frame_rate
        no_face_timestamps.append(start_time)
        start_time = None
        # cv2.imwrite(f'static{i}.jpg', frame)

    video_capture.release()
    return no_face_timestamps

def split_into_parts(input_video, input_audio, timestamps):
    start_time = 0
    current = 1
    to_sync_videos = []
    to_sync_audios = []
    not_to_sync_videos = []
    not_to_sync_audios = []
    for timestamp in tqdm(timestamps):
        if current % 2 == 1:
            to_sync_videos.append(clip_video(dubbed_video, f'EDITS\PARTS\TOSYNC\\to_sync_video{len(to_sync_videos)+1}.mp4', start_time, timestamp))
            to_sync_audios.append(clip_audio(input_audio, f'EDITS\PARTS\TOSYNC\\to_sync_audio{len(to_sync_audios)+1}.wav',
            start_time, timestamp))

            start_time = timestamp
        else:
            not_to_sync_videos.append(clip_video(dubbed_video,f'EDITS\PARTS\\NOTTOSYNC\\not_to_sync_video{len(not_to_sync_videos)+1}.mp4', start_time, timestamp))
            not_to_sync_audios.append(clip_audio(input_audio, f'EDITS\PARTS\\NOTTOSYNC\\not_to_sync_audio{len(not_to_sync_audios)+1}.wav', start_time, timestamp))
            start_time = timestamp
        current += 1
    
    return to_sync_videos, to_sync_audios, not_to_sync_videos, not_to_sync_audios


if __name__ == '__main__':
    input_video_file = r'EDITS\input_video.mp4'
    input_audio_file = r'EDITS\input_audio.wav'
    clipped_video = r'EDITS\clipped_video.mp4'
    dubbed_video = r'EDITS\dubbed_video.mp4'

    print(f'The audio clip is {get_audio_length(input_audio_file)} seconds long.')
    print(f'The video clip is {get_video_length(input_video_file)} seconds long.')

    match_lengths(input_video_file, clipped_video, input_audio_file, input_audio_file)
    replace_audio(clipped_video, input_audio_file, dubbed_video)

    checkpoint_path = r'checkpoints\wav2lip.pth'
    video_path = r'EDITS\dubbed_video.mp4'
    audio_path = r'EDITS\input_audio.wav'
    output_path = r'EDITS\synced_video.mp4'

    # run_inference(checkpoint_path, video_path, audio_path, output_path)

    timestamps = get_timestamps(dubbed_video)
    timestamps.append(get_video_length(dubbed_video))

    for i in timestamps:
        print(i)

    to_sync_videos, to_sync_audios, not_to_sync_videos, not_to_sync_audios = split_into_parts(dubbed_video, input_audio_file,timestamps)

    current = 0
    for i in tqdm(range(0, len(to_sync_videos))):
        run_inference(checkpoint_path, to_sync_videos[i], to_sync_audios[i], f'EDITS\SYNCED\synced{current+1}.mp4')
        current += 1
        if i < len(not_to_sync_videos):
            replace_audio(not_to_sync_videos[i], not_to_sync_audios[i], f'EDITS\SYNCED\synced{current+1}.mp4')
            current += 1
    
    final_clips = []
    for i in tqdm(range(0, current)):
        final_clips.append(f'EDITS\SYNCED\synced{i+1}.mp4')
    
    final_clips = [VideoFileClip(file) for file in final_clips]

    final_clip = concatenate_videoclips(final_clips)
    FINAL_DUB = 'EDITS\FINAL_DUB.mp4'
    final_clip.write_videofile(FINAL_DUB, codec='libx264')