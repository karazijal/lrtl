# %%
import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode, Resize
from tqdm.auto import tqdm


def maybe_slurm_shard(iter_target):
    is_slurm = bool(os.getenv("SLURM_JOB_ID"))
    total = len(iter_target)
    unique_suffix = f"_{os.getpid()}"
    curr_id = 0
    num_ids = 1
    if is_slurm:
        array_id = os.environ.get("SLURM_ARRAY_TASK_ID", None)
        if array_id is not None:
            curr_id = int(array_id)
            unique_suffix += f"_slurm{curr_id}"
            min_id = int(os.environ.get("SLURM_ARRAY_TASK_MIN", 0))
            num_ids = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
            iter_target = iter_target[curr_id - min_id : total : num_ids]
            print(f"Running {curr_id} of {num_ids} jobs, {len(iter_target)} classes")
    return iter_target, is_slurm, unique_suffix, curr_id, num_ids


def load_seq(dataset_path, seq_name, seq_search_path, disable=False, res=(-1, -1)):
    frames = sorted(list((dataset_path / seq_search_path / seq_name).glob("*.jpg")))
    fs = []
    for f in tqdm(frames, disable=disable):
        img = Image.open(f).convert("RGB")
        if res[0] > 0:
            img = Resize(res, interpolation=InterpolationMode.LANCZOS)(img)
        fs.append(np.array(img))
    return np.stack(fs)


def smart_cat(tensor1, tensor2, dim):
    if tensor1 is None:
        return tensor2
    return torch.cat([tensor1, tensor2], dim=dim)


def gen_aux_grid(H, W, F, side=32, time_stride=4):
    grid_width = torch.linspace(0, W - 1, side)
    grid_height = torch.linspace(0, H - 1, side)
    grid_pts = torch.zeros((F // time_stride, side * side, 3))
    grid_pts[:, :, 0] = torch.arange(0, F, time_stride).view(-1, 1).expand(-1, side * side)
    grid_pts[:, :, 1] = grid_width.repeat(len(grid_height)).expand(F // time_stride, -1)
    grid_pts[:, :, 2] = grid_height.repeat_interleave(len(grid_width)).expand(F // time_stride, -1)
    return grid_pts.view(1, -1, 3)


def gen_aux_grid(H, W, QF, F, side=32, time_stride=3):
    grid_width = torch.linspace(0, W - 1, side)
    grid_height = torch.linspace(0, H - 1, side)
    ran = torch.arange(
        QF - 7,
        QF + 7 + 1,
    ).clamp(0, F - 1)
    lb = ran.min()
    ub = ran.max()
    ran = torch.tensor([lb, ub])
    ran = ran[ran != QF]
    n = len(ran)
    grid_pts = torch.zeros((n, side * side, 3))
    print(ran)
    if n == 0:
        return grid_pts
    grid_pts[:, :, 0] = ran.view(-1, 1).expand(-1, side * side)
    grid_pts[:, :, 1] = grid_width.repeat(len(grid_height)).expand(n, -1)
    grid_pts[:, :, 2] = grid_height.repeat_interleave(len(grid_width)).expand(n, -1)
    return grid_pts.view(1, -1, 3)


@torch.no_grad()
def compute_dense_tracks(
    cotracker,
    video,
    grid_query_frame,
    grid_step=16,
    grid_stride=1,
    backward_tracking=True,
    tar_path="frame",
    disable=False,
    checkpoint_freq=20,
    max_frames=-1,
    aux_grid_size=32,
):
    b, nframes, *_, H, W = video.shape
    # grid_width = (W // grid_step) // grid_stride
    # grid_height = (H // grid_step) // grid_stride
    grid_offset_x, grid_offset_y = [(ox, oy) for oy in range(0, grid_stride) for ox in range(0, grid_stride)][
        grid_query_frame % (grid_stride * grid_stride)
    ]
    grid_offsets = (grid_offset_x, grid_offset_y)
    grid_width = torch.arange(grid_offset_x, W, grid_stride, device=video.device)
    grid_height = torch.arange(grid_offset_y, H, grid_stride, device=video.device)
    npoints = grid_width.shape[0] * grid_height.shape[0]
    print(
        f"Grid R^({grid_height.shape[0]}x{grid_width.shape[0]}) \in [{grid_height[0].item(), grid_height[-1].item()}] x [{grid_width[0].item(), grid_width[-1].item()}] for {npoints} points"
    )

    tracks = visibilities = points = None
    aux_tracks = aux_visibilities = aux_points = None
    checkpoint_step = -1
    if max_frames > 0 and nframes > max_frames:

        context_len = max_frames // 2
        new_grid_query_frame = context_len
        start_index = grid_query_frame - context_len
        end_index = grid_query_frame + context_len + 1
        pad_before = 0
        if start_index < 0:
            pad_before = -start_index
            start_index = 0
        pad_after = 0
        if end_index > nframes:
            pad_after = end_index - nframes
            end_index = nframes
        if pad_before > 0:
            end_index += pad_before
            new_grid_query_frame = grid_query_frame
        if pad_after > 0:
            start_index -= pad_after
            new_grid_query_frame += pad_after

        print(
            f"Shrunk video to {max_frames} frames {start_index}:{end_index} with query frame {new_grid_query_frame} from {grid_query_frame}"
        )
        assert start_index >= 0
        assert end_index <= nframes
        assert end_index - start_index == max_frames + 1
        assert new_grid_query_frame >= 0
        assert new_grid_query_frame <= max_frames

        grid_query_frame = new_grid_query_frame
        video = video[:, start_index:end_index].clone()

    checkpoint_path = tar_path.parent / f"{tar_path.stem}.checkpoint.pt"
    if checkpoint_path.exists():
        print(f"Found checkpoint {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=video.device)
        except IOError:
            print(f"File {checkpoint_path} is corrupted; removing")
            checkpoint_path.unlink()
        tracks = checkpoint["tracks"]
        visibilities = checkpoint["visibilities"]
        points = checkpoint["points"]

        aux_tracks = checkpoint["aux_tracks"]
        aux_visibilities = checkpoint["aux_visibilities"]
        aux_points = checkpoint["aux_points"]

        assert grid_step == checkpoint["grid_step"]
        assert grid_stride == checkpoint.get("grid_stride", 1)
        assert max_frames == checkpoint.get("max_frames", -1) or nframes < max_frames
        assert aux_grid_size == checkpoint.get("aux_grid_size", 32)
        checkpoint_step = checkpoint["checkpoint_step"]

    # grid_pts = torch.zeros((1, grid_width * grid_height, 3)).to(video.device)
    # grid_pts[0, :, 0] = grid_query_frame
    nframes = video.shape[1]
    aux_grid = gen_aux_grid(H, W, grid_query_frame, nframes, side=aux_grid_size, time_stride=2).to(video.device)
    print(f"{aux_grid.shape=}")

    offsets = [(ox, oy) for oy in range(0, grid_step) for ox in range(0, grid_step)]
    for current_id, (ox, oy) in enumerate(tqdm(offsets, disable=disable or len(offsets) == 1)):
        if current_id <= checkpoint_step:
            continue
        grid_x = grid_width[ox::grid_step]
        grid_y = grid_height[oy::grid_step]
        npts = len(grid_x) * len(grid_y)
        grid_pts = torch.zeros((1, npts, 3)).to(video.device)
        grid_pts[0, :, 0] = grid_query_frame
        grid_pts[0, :, 1] = grid_x.repeat(len(grid_y))
        grid_pts[0, :, 2] = grid_y.repeat_interleave(len(grid_x))
        # print(f"{grid_pts.shape=}")
        q_grid_pts = torch.cat([grid_pts, aux_grid], dim=1)
        tracks_step, visibilities_step = cotracker._compute_sparse_tracks(
            video=video,
            queries=q_grid_pts,
            backward_tracking=backward_tracking,
        )
        # print(f"{tracks_step.shape=}")

        aux_track_step = tracks_step[:, :, npts:].clone()
        aux_vis_step = visibilities_step[:, :, npts:].clone()

        tracks_step = tracks_step[:, :, :npts].clone()
        visibilities_step = visibilities_step[:, :, :npts].clone()

        tracks = smart_cat(tracks, tracks_step, dim=2)
        visibilities = smart_cat(visibilities, visibilities_step, dim=2)
        points = smart_cat(points, grid_pts, dim=1)

        aux_tracks = smart_cat(aux_tracks, aux_track_step, dim=2)
        aux_visibilities = smart_cat(aux_visibilities, aux_vis_step, dim=2)
        aux_points = smart_cat(aux_points, aux_grid, dim=1)

        if disable and current_id % 10 == 0:
            print(f"[{current_id+1:03d}/{grid_step*grid_step:03d}] {tracks.shape[2]} tracks")
        if disable and len(offsets) == 1:
            print(f"{tracks.shape[2]} tracks")

        if current_id % checkpoint_freq == 0 and len(offsets) > 1:
            torch_save_atomic(
                {
                    "tracks": tracks,
                    "visibilities": visibilities,
                    "points": points,
                    "aux_tracks": aux_tracks,
                    "aux_visibilities": aux_visibilities,
                    "aux_points": aux_points,
                    "grid_step": grid_step,
                    "grid_stride": grid_stride,
                    "aux_grid_size": aux_grid_size,
                    "checkpoint_step": current_id,
                    "max_frames": max_frames,
                },
                checkpoint_path,
            )
            print(f"Saved checkpoint {checkpoint_path}")
            checkpoint_step = current_id

    return (
        grid_offsets,
        tracks,
        visibilities,
        points,
        aux_tracks,
        aux_visibilities,
        aux_points,
        grid_query_frame,
        video.shape[1],
    )


def torch_save_atomic(data, path):
    tmp_path = path.parent / f"{path.stem}.{os.getpid()}.{os.uname()[1]}"
    torch.save(data, tmp_path)
    tmp_path.replace(path)  # on POSIX, this is atomic but will overwrite existing files (os.rename is the same)


def frame_path_to_fid(frame_path):
    seq_name = frame_path.parent.name
    stem = frame_path.stem
    last = stem.split("_")[-1]
    last = last.replace(seq_name, "").replace("Comp", "").replace("frame", "").lstrip("_").lstrip("0")
    try:
        fid = int(last) if last else 0
    except ValueError:
        print("Cannot parse frame path fid", frame_path)
        raise
    return fid


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda", help="Device (cpu or cuda)")
    parser.add_argument(
        "--check-integrety", action="store_true", help="Check integrity of existing files", default=False
    )
    parser.add_argument(
        "--precheck", action="store_true", help="Precheck if files exists before sharding", default=False
    )

    parser.add_argument("--grid_step", type=int, default=1, help="Grid size")
    parser.add_argument("--grid_stride", type=int, default=4, help="Grid stride")
    parser.add_argument("--max_frames", type=int, default=100, help="Max frames")
    parser.add_argument("--height", type=int, default=480, help="Vid height")
    parser.add_argument("--width", type=int, default=854, help="Vid width")

    parser.add_argument("--frame-suffix", type=str, default=".jpg", help="Frame suffix")
    parser.add_argument("--seq-search-path", type=str, default="JPEGImages/480p")
    parser.add_argument("--seqs", type=str, nargs="*", help="Sequences to process")

    parser.add_argument("--frame", type=str, default=None, help="name")
    parser.add_argument("--rawout", action="store_true", help="Save raw output", default=False)

    parser.add_argument("dataset_dir", type=Path, help="Path to dataset directory")
    parser.add_argument("output_dir", type=Path, help="Path to output directory")

    args = parser.parse_args()

    output_path = args.output_dir
    output_path.mkdir(exist_ok=True)

    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker2")
    cotracker = cotracker.to(args.device)
    cotracker.eval()

    if args.seqs:
        seqs = []
        for seq in args.seqs:
            s = seq.strip()
            s = args.dataset_dir / args.seq_search_path / s
            if s.is_dir():
                seqs.append(s.name)
            else:
                print(f"WARNING: Sequence {s} not found")
    else:
        seqs = sorted([l.name for l in (args.dataset_dir / args.seq_search_path).iterdir() if l.is_dir()])

    iter_target = []
    for seq in seqs:
        frames = sorted(
            [p for p in (args.dataset_dir / args.seq_search_path / seq).glob(f"*{args.frame_suffix}") if p.is_file()]
        )
        frames = [(fid, f) for fid, f in enumerate(frames)]

        if len(frames) > 0:
            if args.frame:
                new_frames = [f for f in frames if args.frame == f[1].name]
                if len(new_frames) != 1:
                    raise ValueError(f"Frame {args.frame} not found in sequence {seq}")
                iter_target.extend(new_frames)
            else:
                iter_target.extend(frames)
            test_img = np.array(Image.open(frames[0][1]).convert("RGB"))
            print(
                f"Seq {seq} should have frames of shape {(len(frames), *test_img.shape)} frame pattern {frames[0][1]}"
            )
            decoded_fid = frame_path_to_fid(frames[0][1])
            if decoded_fid != frames[0][0]:
                print(f"WARNING: Sequence {seq} starts at frame {decoded_fid} not {frames[0][0]}")
        else:
            print(f"WARNING: Sequence {seq} has no frames")
    print(f"Found {len(seqs)} sequences with total of {len(iter_target)} frames")

    if args.precheck:
        new_iter_target = []
        for fid, frame_path in tqdm(iter_target):
            seq = frame_path.parent.name
            frame_name = frame_path.stem
            tar_path = output_path / f"{seq}-{frame_name}.pt"
            if not tar_path.exists():
                new_iter_target.append((fid, frame_path))
        iter_target = new_iter_target
        print(f"Filterred down to {len(iter_target)} frames")

    iter_target, is_slurm, _, curr_id, num_ids = maybe_slurm_shard(iter_target)
    if is_slurm:
        random.shuffle(iter_target)

    for job_id, (fid, frame_path) in enumerate(tqdm(iter_target, disable=is_slurm)):
        frame_name = frame_path.stem
        seq = frame_path.parent.name
        tar_path = output_path / f"{seq}-{frame_name}.pt"

        if tar_path.exists():
            if args.check_integrety:
                try:
                    torch.load(tar_path)
                    print(f"File {tar_path} already exists; skipping")
                    continue
                except (IOError, EOFError, OSError):
                    print(f"File {tar_path} is corrupted; overwriting")
                    tar_path.unlink()
            else:
                print(f"File {tar_path} already exists; skipping")
                continue

        if is_slurm:
            print(
                f"[{curr_id:02d} / {num_ids:02d}] Processing {seq}-{frame_name} ({fid}) ({job_id+1:03d}/{len(iter_target):04d})"
            )

        # Process a tensor:
        frames = load_seq(args.dataset_dir, seq, args.seq_search_path, disable=True, res=(args.height, args.width))
        nframes, height, width, _ = frames.shape
        frames_tensor = torch.from_numpy(frames).float().permute(0, 3, 1, 2).float().unsqueeze(0).to(args.device)

        offset, t, v, p, at, av, ap, query_frame, new_nframes = compute_dense_tracks(
            cotracker,
            frames_tensor,
            grid_query_frame=fid,
            grid_step=args.grid_step,
            grid_stride=args.grid_stride,
            backward_tracking=True,
            tar_path=tar_path,
            disable=is_slurm,
            max_frames=args.max_frames,
        )
        t = t.squeeze(0).float().cpu().contiguous()
        v = v.squeeze(0).bool().cpu().contiguous()
        p = p.squeeze(0).float().cpu().contiguous()

        at = at.squeeze(0).float().cpu().contiguous()
        av = av.squeeze(0).bool().cpu().contiguous()
        ap = ap.squeeze(0).float().cpu().contiguous()

        print(f"{t.shape=} {v.shape=} {p.shape=} {at.shape=} {av.shape=} {ap.shape=}")

        if args.rawout:
            torch_save_atomic(
                {
                    "t": t,
                    "v": v,
                    "p": p,
                    "at": at,
                    "av": av,
                    "ap": ap,
                },
                Path((str(tar_path) + "_rawout")),
            )

        x = t[query_frame, :, 0].round().long()
        y = t[query_frame, :, 1].round().long()
        dest_tracks = torch.zeros(new_nframes, height, width, 2, dtype=torch.float32).float()
        dest_vis = torch.zeros(new_nframes, height, width, dtype=torch.bool).bool()

        dest_tracks[:, y, x] = t[:, torch.arange(x.shape[0])]
        dest_vis[:, y, x] = v[:, torch.arange(x.shape[0])]

        dest_tracks = dest_tracks[:, offset[1] :: args.grid_stride, offset[0] :: args.grid_stride].half().clone()
        dest_vis = dest_vis[:, offset[1] :: args.grid_stride, offset[0] :: args.grid_stride].bool().clone()

        torch_save_atomic(
            {
                "tracks": dest_tracks,
                "vis": dest_vis,
                "seq": seq,
                "fid": query_frame,
                "height": height,
                "width": width,
                "offset": offset,
            },
            tar_path,
        )

        torch_save_atomic(
            {
                "tracks": at.half().clone(),
                "vis": av.bool().clone(),
                "points": ap.half().clone(),
                "seq": seq,
                "fid": query_frame,
                "height": height,
                "width": width,
                "offset": offset,
            },
            tar_path.parent / f"{tar_path.stem}_aux.pt",
        )


if __name__ == "__main__":
    main()

# %%
