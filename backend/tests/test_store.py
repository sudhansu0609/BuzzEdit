from backend.store import ProjectStore

def test_project_store_atomic_save_and_load(tmp_path):
    store = ProjectStore(base_dir=str(tmp_path / "projects"))
    
    proj_id = "test_p1"
    data = {"id": proj_id, "name": "Test Project", "source_video": "video.mp4"}
    
    store.save_project(proj_id, data)
    
    loaded = store.get_project(proj_id)
    assert loaded is not None
    assert loaded["name"] == "Test Project"

    projects = store.list_projects()
    assert len(projects) == 1
    assert projects[0]["id"] == proj_id

    deleted = store.delete_project(proj_id)
    assert deleted is True
    assert store.get_project(proj_id) is None
