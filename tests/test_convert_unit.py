"""Unit tests on synthetic data (no dataset needed)."""
from malaria_ssl.data.convert import build_coco, orientation_check, source_bbox_to_coco, to_binary
from malaria_ssl.data.splits import stratified_split

CATS = [{"id": 1, "name": "red blood cell"}, {"id": 2, "name": "ring"}, {"id": 3, "name": "difficult"}]


def _rec(name, h, w, objs):
    return {"image": {"pathname": f"/images/{name}", "shape": {"r": h, "c": w, "channels": 3}},
            "objects": [{"category": c, "bounding_box": {"minimum": {"r": r0, "c": c0},
                                                 "maximum": {"r": r1, "c": c1}}}
                        for c, (r0, c0, r1, c1) in objs]}


def test_bbox_row_col_to_xywh():
    # rows are y, columns are x
    bbox = {"minimum": {"r": 10, "c": 200}, "maximum": {"r": 50, "c": 260}}
    assert source_bbox_to_coco(bbox) == [200.0, 10.0, 60.0, 40.0]


def test_build_coco_and_orientation():
    # tall-narrow image: H=100, W=300; box spans columns 250..290 (valid only if c=x)
    recs = [_rec("a.png", 100, 300, [("ring", (10, 250, 40, 290))])]
    coco = build_coco(recs, CATS, images_dir=".", image_ids={"a.png": 7}, check_files=False)
    assert coco["images"] == [{"id": 7, "file_name": "a.png", "width": 300, "height": 100}]
    ann = coco["annotations"][0]
    assert ann["bbox"] == [250.0, 10.0, 40.0, 30.0] and ann["category_id"] == 2
    assert ann["area"] == 1200.0
    oc = orientation_check(recs)
    assert oc["violations_r_is_row"] == 0 and oc["violations_r_is_col"] == 1


def test_to_binary_drops_mapped_none():
    recs = [_rec("a.png", 100, 100, [("red blood cell", (0, 0, 5, 5)), ("ring", (0, 0, 5, 5)),
                                     ("difficult", (0, 0, 5, 5))])]
    coco = build_coco(recs, CATS, ".", {"a.png": 1}, check_files=False)
    mapping = {"red blood cell": "uninfected", "ring": "infected", "difficult": None}
    bcats = [{"id": 1, "name": "uninfected"}, {"id": 2, "name": "infected"}]
    out, dropped = to_binary(coco, mapping, bcats)
    assert dropped == {"difficult": 1}
    assert [a["category_id"] for a in out["annotations"]] == [1, 2]


def test_stratified_split_rare_class_in_val_and_groups():
    labels = {i: {1} for i in range(1, 101)}
    for i in (3, 30, 60, 90):
        labels[i] = {1, 2}  # rare class in 4 images
    train, val = stratified_split(labels, 0.25, seed=0)
    assert not set(train) & set(val) and len(train) + len(val) == 100
    assert any(2 in labels[i] for i in val) and any(2 in labels[i] for i in train)
    assert abs(len(val) - 25) <= 1
    assert stratified_split(labels, 0.25, seed=0) == (train, val)  # deterministic
    groups = {i: f"g{i // 10}" for i in labels}
    train, val = stratified_split(labels, 0.25, seed=0, groups=groups)
    assert not {groups[i] for i in train} & {groups[i] for i in val}
