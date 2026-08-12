from fastapi import APIRouter, HTTPException
from models import RenderJob, Clip, ClipType, Project
from config import PROJECTS_DIR, OUTPUT_DIR
import asyncio
import uuid

router = APIRouter()
jobs: dict = {}


@router.post("/render")
async def render_video(job: RenderJob):
    from routes.projects import projects

    if job.project_id not in projects:
        raise HTTPException(status_code=404, detail="Project not found")

    project = projects[job.project_id]

    if not project.source_video:
        raise HTTPException(status_code=400, detail="No source video uploaded for project")


    job_id = f"render_{job.project_id}"
    output_path = job.output_path or str(OUTPUT_DIR / f"{job.project_id}_output.mp4")

    jobs[job_id] = {
        "id": job_id,
        "project_id": job.project_id,
        "job_type": "render",
        "status": "running",
        "progress": 0.0,
    }

    try:
        result_path = await asyncio.to_thread(
            run_render,
            project,
            output_path,
            job.settings or project.settings,
        )

        project.output_path = result_path
        project.status = "rendered"

        project_dir = PROJECTS_DIR / job.project_id
        (project_dir / "project.json").write_text(project.model_dump_json(indent=2))

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 1.0
        jobs[job_id]["result"] = {"output_path": result_path}

        return {
            "status": "completed",
            "job_id": job_id,
            "output_path": result_path,
        }

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))


def run_render(project: Project, output_path: str, settings: dict) -> str:
    from video_pipeline.cutter import cut_video_segments
    from video_pipeline.exporter import export_final_video

    good_segments = get_good_segments(project)

    clip_files = cut_video_segments(
        project.source_video,
        good_segments,
        project_dir=PROJECTS_DIR / project.id,
    )

    transition_type = settings.get("transition_type", "crossfade")
    transition_duration = settings.get("transition_duration", 0.5)

    resolution = settings.get("resolution", "1920x1080")
    width, height = map(int, resolution.split("x"))

    export_final_video(
        clip_files=clip_files,
        output_path=output_path,
        width=width,
        height=height,
        fps=settings.get("fps", 30),
        transition_type=transition_type,
        transition_duration=transition_duration,
    )

    return output_path


def get_good_segments(project: Project) -> list[tuple[float, float]]:
    if project.clips:
        return [(c.start_time, c.end_time) for c in project.clips if c.clip_type != ClipType.SILENCE and c.clip_type != ClipType.FUMBLE]

    if not project.detected_segments or not project.transcript:
        from utils.ffmpeg_utils import get_video_duration
        duration = get_video_duration(project.source_video)
        return [(0.0, duration)]

    cut_ranges = [(s.start, s.end) for s in project.detected_segments]
    total_duration = project.transcript.duration

    good_segments = []
    current = 0.0

    for start, end in sorted(cut_ranges, key=lambda x: x[0]):
        if start > current:
            good_segments.append((current, start))
        current = max(current, end)

    if current < total_duration:
        good_segments.append((current, total_duration))

    return good_segments


@router.get("/status/{job_id}")
async def render_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@router.post("/{project_id}/auto_edit")
async def auto_edit(project_id: str):
    from routes.projects import projects

    if project_id not in projects:
        raise HTTPException(status_code=404, detail="Project not found")

    project = projects[project_id]
    results = {}

    if project.status == "draft":
        from routes.transcription import transcribe_video
        from models import TranscribeJob
        tr_job = TranscribeJob(project_id=project_id)
        results["transcription"] = await transcribe_video(tr_job)

    if project.status in ("transcribed", "analyzed"):
        from routes.analysis import analyze_video
        from models import AnalyzeJob
        an_job = AnalyzeJob(
            project_id=project_id,
            remove_silence=project.settings.get("remove_silence", True),
            remove_fumbles=project.settings.get("remove_fumbles", True),
        )
        results["analysis"] = await analyze_video(an_job)

    render_job = RenderJob(project_id=project_id)
    results["render"] = await render_video(render_job)

    return results
