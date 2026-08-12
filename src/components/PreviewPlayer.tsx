import { useRef, useEffect, useCallback, useState } from 'react';
import { useProjectStore } from '../hooks/store';

export default function PreviewPlayer() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const { project, currentTime, setCurrentTime, isPlaying, setIsPlaying, selectedClipId } = useProjectStore();
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(false);
  const [videoError, setVideoError] = useState<string | null>(null);
  const [videoReady, setVideoReady] = useState(false);

  const sourceVideo = project?.source_video || project?.sourceVideo;

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !sourceVideo) return;

    const handleTimeUpdate = () => setCurrentTime(video.currentTime);
    const handlePlay = () => setIsPlaying(true);
    const handlePause = () => setIsPlaying(false);
    const handleEnded = () => setIsPlaying(false);
    const handleLoadedData = () => setVideoReady(true);
    const handleError = () => setVideoError('Failed to stream video');

    video.addEventListener('timeupdate', handleTimeUpdate);
    video.addEventListener('play', handlePlay);
    video.addEventListener('pause', handlePause);
    video.addEventListener('ended', handleEnded);
    video.addEventListener('loadeddata', handleLoadedData);
    video.addEventListener('error', handleError);

    // Stream video through backend HTTP Range endpoint
    video.src = `http://localhost:8099/api/media/stream?path=${encodeURIComponent(sourceVideo)}`;
    video.load();

    return () => {
      video.removeEventListener('timeupdate', handleTimeUpdate);
      video.removeEventListener('play', handlePlay);
      video.removeEventListener('pause', handlePause);
      video.removeEventListener('ended', handleEnded);
      video.removeEventListener('loadeddata', handleLoadedData);
      video.removeEventListener('error', handleError);
    };
  }, [sourceVideo, setCurrentTime, setIsPlaying]);

  useEffect(() => {
    if (videoRef.current && Math.abs(videoRef.current.currentTime - currentTime) > 0.3) {
      videoRef.current.currentTime = currentTime;
    }
  }, [currentTime]);

  const togglePlay = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    if (isPlaying) {
      video.pause();
    } else {
      video.play().catch(() => {});
    }
  }, [isPlaying]);

  const handleSeek = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const video = videoRef.current;
    if (!video || !video.duration) return;
    const newTime = x * video.duration;
    video.currentTime = newTime;
    setCurrentTime(newTime);
  }, [setCurrentTime]);

  const toggleMute = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    video.muted = !isMuted;
    setIsMuted(!isMuted);
  }, [isMuted]);

  const handleVolumeChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const video = videoRef.current;
    if (!video) return;
    const vol = parseFloat(e.target.value);
    video.volume = vol;
    setVolume(vol);
    if (vol === 0) setIsMuted(true);
  }, []);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const duration = videoRef.current?.duration || 60;

  return (
    <div className="preview-player">
      <div className="video-container">
        {sourceVideo ? (
          <>
            <video
              ref={videoRef}
              className="video-element"
              onClick={togglePlay}
            />
            {!videoReady && !videoError && (
              <div className="video-loading">
                <div className="spinner" />
                <span>Loading video stream...</span>
              </div>
            )}
            {videoError && (
              <div className="video-error">
                <span>{videoError}</span>
              </div>
            )}
          </>
        ) : (
          <div className="video-placeholder">
            <span>No video loaded</span>
          </div>
        )}
      </div>

      <div className="player-controls">
        <div className="seek-bar" onClick={handleSeek}>
          <div
            className="seek-progress"
            style={{ width: `${(currentTime / duration) * 100}%` }}
          />
        </div>

        <div className="controls-row">
          <button className="btn btn-icon" onClick={togglePlay}>
            {isPlaying ? '⏸' : '▶'}
          </button>

          <span className="time-display font-mono text-xs">
            {formatTime(currentTime)} / {formatTime(duration)}
          </span>

          <div className="volume-control">
            <button className="btn btn-icon" onClick={toggleMute}>
              {isMuted ? '🔇' : volume < 0.5 ? '🔉' : '🔊'}
            </button>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={volume}
              onChange={handleVolumeChange}
              className="volume-slider"
            />
          </div>
        </div>
      </div>
    </div>
  );
}
