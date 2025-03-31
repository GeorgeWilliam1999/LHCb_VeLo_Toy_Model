from LHCB_Velo_Toy_Models.state_event_generator import *

class EventValidator:
    def __init__(self, truth_event: Event, rec_tracks: list[Track]):
        self.truth_event = truth_event
        self.rec_tracks = rec_tracks
        self.truth_tracks = {track.track_id: track for track in truth_event.tracks}
        self.truth_track_hits = {
            track.track_id: {hit.hit_id for hit in track.hits} 
            for track in truth_event.tracks
        }
        self.rec_track_hits = {
            track.track_id: {hit.hit_id for hit in track.hits} 
            for track in rec_tracks
        }
    
    def match_tracks(self, threshold: float = 0.7):
        """
        For each reconstructed track, find the truth track that gives the highest
        hit-overlap fraction (i.e. fraction of rec track hits that are in the truth track).
        If that fraction meets or exceeds the threshold, record the match.
        """
        match_results = {}  # rec_track_id -> (overlap_fraction, truth_track_id or None)
        truth_matches = {}  # truth_track_id -> list of rec_track_ids matched
        
        for rec_id, rec_hits in self.rec_track_hits.items():
            best_match = None
            best_fraction = 0.0
            for truth_id, truth_hits in self.truth_track_hits.items():
                if len(rec_hits) == 0:
                    continue
                overlap_fraction = len(rec_hits.intersection(truth_hits)) / len(rec_hits)
                if overlap_fraction > best_fraction:
                    best_fraction = overlap_fraction
                    best_match = truth_id
            if best_fraction >= threshold:
                match_results[rec_id] = (best_fraction, best_match)
                truth_matches.setdefault(best_match, []).append(rec_id)
            else:
                match_results[rec_id] = (best_fraction, None)
        return match_results, truth_matches

    def compute_metrics(self, threshold: float = 0.7):
        match_results, truth_matches = self.match_tracks(threshold)
        total_rec_tracks = len(self.rec_tracks)
        
        # Ghost tracks: rec tracks with no valid match.
        ghost_tracks = [rec_id for rec_id, (frac, truth_id) in match_results.items() if truth_id is None]
        ghost_rate = len(ghost_tracks) / total_rec_tracks if total_rec_tracks > 0 else 0.0
        
        # Clones: additional rec tracks for a truth track.
        clones_count = sum(len(rec_ids) - 1 for rec_ids in truth_matches.values() if len(rec_ids) > 1)
        clone_fraction = clones_count / total_rec_tracks if total_rec_tracks > 0 else 0.0
        
        total_truth_tracks = len(self.truth_tracks)
        matched_truth_tracks = len(truth_matches)
        reconstruction_efficiency = matched_truth_tracks / total_truth_tracks if total_truth_tracks > 0 else 0.0
        
        # Hit efficiency: fraction of truth hits that are associated with at least one rec track.
        matched_rec_hit_ids = set()
        for rec_id, (frac, truth_id) in match_results.items():
            if truth_id is not None:
                matched_rec_hit_ids.update(self.rec_track_hits[rec_id])
        total_truth_hits = sum(len(hits) for hits in self.truth_track_hits.values())
        hit_efficiency = len(matched_rec_hit_ids) / total_truth_hits if total_truth_hits > 0 else 0.0
        
        # Purity: average overlap fraction among rec tracks with a valid truth match.
        purity_values = [frac for rec_id, (frac, truth_id) in match_results.items() if truth_id is not None]
        average_purity = np.mean(purity_values) if purity_values else 0.0
        
        metrics = {
            'ghost_rate': ghost_rate,
            'clone_fraction': clone_fraction,
            'reconstruction_efficiency': reconstruction_efficiency,
            'hit_efficiency': hit_efficiency,
            'purity': average_purity,
            'total_rec_tracks': total_rec_tracks,
            'total_truth_tracks': total_truth_tracks,
            'ghost_tracks': ghost_tracks,
            'clones_count': clones_count
        }
        return metrics

    def print_metrics(self, threshold: float = 0.7):
        metrics = self.compute_metrics(threshold)
        print("=== Event Validation Metrics ===")
        print(f"Total Reconstructed Tracks: {metrics['total_rec_tracks']}")
        print(f"Total Truth Tracks: {metrics['total_truth_tracks']}")
        print(f"Reconstruction Efficiency: {metrics['reconstruction_efficiency']*100:.2f}%")
        print(f"Ghost Rate: {metrics['ghost_rate']*100:.2f}%")
        print(f"Clone Fraction: {metrics['clone_fraction']*100:.2f}% ({metrics['clones_count']} clones)")
        print(f"Hit Efficiency: {metrics['hit_efficiency']*100:.2f}%")
        print(f"Purity: {metrics['purity']*100:.2f}%")
