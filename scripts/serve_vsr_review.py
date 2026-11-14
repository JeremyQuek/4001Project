"""Serve a local human-review worksheet and save decisions atomically."""
import argparse
import hashlib
import json
import re
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
STATUSES = {'unreviewed','approve','original_only','set_aside'}
REASONS = {'objects_unclear','label_questionable','spatial_ambiguity','reference_frame','transformation','other'}


class ReviewError(ValueError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


class ReviewStore:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.folder = self.root/'data/vsr/review'
        self.dataset = json.loads((self.folder/'dataset.json').read_text(encoding='utf-8'))
        self.rows = {r['example_id']:r for r in self.dataset['rows']}
        self.path = self.folder/'human_decisions.json'
        self.lock = threading.RLock()
        if self.path.exists():
            self.state = json.loads(self.path.read_text(encoding='utf-8'))
            if self.state['dataset_fingerprint'] != self.dataset['dataset_fingerprint']:
                raise ReviewError('Dataset changed. Preserve the old review and rebuild a fresh review workspace.')
        else:
            self.state = {'dataset_fingerprint':self.dataset['dataset_fingerprint'],
                'revision':0,'reviewer':'','decisions':{},'greenlight':None}

    def snapshot(self):
        return json.loads(json.dumps(self.state))

    def counts(self):
        counts = {status:0 for status in STATUSES}
        for example_id in self.rows:
            counts[self.state['decisions'].get(example_id,{}).get('status','unreviewed')] += 1
        return counts

    def validate_request(self, payload):
        if payload.get('dataset_fingerprint') != self.dataset['dataset_fingerprint']:
            raise ReviewError('This review belongs to another dataset. Reload the page.')
        if payload.get('expected_revision') != self.state['revision']:
            raise ReviewError('Review changed in another tab. Reload before saving again.')

    def save(self, payload):
        with self.lock:
            self.validate_request(payload)
            reviewer = payload.get('reviewer', self.state['reviewer'])
            if not isinstance(reviewer,str) or len(reviewer)>100:
                raise ReviewError('Reviewer name must be under 100 characters.')
            example_id = payload.get('example_id')
            if example_id is not None:
                if example_id not in self.rows:
                    raise ReviewError('Unknown statement.')
                decision = payload.get('decision',{})
                status = decision.get('status')
                reasons = decision.get('reasons',[])
                note = decision.get('note','')
                if status not in STATUSES or not isinstance(reasons,list) or any(r not in REASONS for r in reasons):
                    raise ReviewError('Invalid decision or reason.')
                if not isinstance(note,str) or len(note)>2000:
                    raise ReviewError('Notes must be under 2,000 characters.')
                if status in ('original_only','set_aside') and not reasons:
                    raise ReviewError('Choose a reason for Original only or Set aside.')
                if status=='approve' and reasons:
                    raise ReviewError('Clear issue reasons before approving all checks, or choose Original only / Set aside.')
                if status=='original_only' and any(r in reasons for r in ('objects_unclear','label_questionable')):
                    raise ReviewError('Use Set aside when the original objects or source label are questionable.')
                if 'other' in reasons and not note.strip():
                    raise ReviewError('Add a short note when choosing Other.')
                clean = {'status':status,'reasons':sorted(set(reasons)),'note':note.strip(),'updated_at':now()}
                self.state['decisions'][example_id] = clean
            self.state['reviewer'] = reviewer.strip()
            self.state['revision'] += 1
            self.state['greenlight'] = None
            atomic_json(self.path,self.state)
            return self.snapshot()

    def greenlight(self, payload):
        with self.lock:
            self.validate_request(payload)
            if not payload.get('confirm'):
                raise ReviewError('Confirm that the same review criteria were applied throughout.')
            if not self.state['reviewer']:
                raise ReviewError('Enter your reviewer name before greenlighting.')
            if self.counts()['unreviewed']:
                raise ReviewError('Review every statement before greenlighting.')
            if not (self.counts()['approve']+self.counts()['original_only']):
                raise ReviewError('At least one statement must be included for extraction.')
            for split in ('train','validation','test'):
                if not any(r['split']==split and self.state['decisions'].get(r['example_id'],{}).get('status') in ('approve','original_only') for r in self.dataset['rows']):
                    raise ReviewError(f'No included {split} examples remain. Recheck the review before greenlighting.')
            # Re-check source statements and image bytes before creating approved manifests.
            source = self.root/'data/vsr/processed/directional_candidates.jsonl'
            if hashlib.sha256(source.read_bytes()).hexdigest() != self.dataset['candidate_sha256']:
                raise ReviewError('Source metadata changed. Rebuild and review the changed dataset.')
            for image_id, info in self.dataset['images'].items():
                path = self.root/f'data/vsr/images/{image_id}.jpg'
                if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != info['sha256']:
                    raise ReviewError(f'Image {image_id} changed or is missing. Restore it before greenlighting.')
            approved, original_only, aside, all_rows = [], [], [], []
            for row in self.dataset['rows']:
                decision = self.state['decisions'][row['example_id']]
                result = dict(row, human_review=decision, consistency_eligible=decision['status']=='approve')
                result['variants'] = dict(row['variants'])
                if result['consistency_eligible']:
                    result['variants']['converse_truth'] = row['truth_label']
                    result['variants']['mirrored_equivalent_truth'] = row['truth_label']
                all_rows.append(result)
                {'approve':approved,'original_only':original_only,'set_aside':aside}[decision['status']].append(result)
            selected = approved+original_only
            destination = self.folder/f'approved/revision_{self.state["revision"]:05}'
            destination.mkdir(parents=True,exist_ok=True)
            artifacts = {}
            subsets = {'reviewed_all':all_rows,'extraction_manifest':selected,
                'consistency_manifest':approved,'set_aside':aside}
            subsets.update({split:[r for r in selected if r['split']==split] for split in ('train','validation','test')})
            for name, rows in subsets.items():
                content = ''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
                path = destination/f'{name}.jsonl'
                path.write_bytes(content.encode('utf-8'))
                artifacts[name] = {'path':str(path.relative_to(self.root)),
                    'rows':len(rows),'sha256':hashlib.sha256(content.encode()).hexdigest()}
            approval = {'approved_at':now(),'reviewer':self.state['reviewer'],
                'dataset_fingerprint':self.dataset['dataset_fingerprint'],'review_revision':self.state['revision'],
                'counts':self.counts(),'artifacts':artifacts,
                'truth_labels_changed':0,'model_predictions_shown':False,
                'scope':'Greenlight reviewed manifests for feature extraction. No GPU job launched.'}
            atomic_json(destination/'approval.json',approval)
            self.state['greenlight'] = approval
            atomic_json(self.path,self.state)
            return self.snapshot()


def handler_class(store, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, status, content, content_type='application/json; charset=utf-8', disposition=None):
            body = content if isinstance(content,bytes) else json.dumps(content).encode()
            self.send_response(status)
            self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            if disposition:
                self.send_header('Content-Disposition',disposition)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            route = urlparse(self.path).path
            if route=='/api/dataset':
                return self.send(200,store.dataset)
            if route in ('/api/state','/api/export'):
                with store.lock:
                    value = store.snapshot()
                return self.send(200,value,disposition='attachment; filename="vsr-human-review.json"' if route.endswith('export') else None)
            if re.fullmatch(r'/images/\d{12}\.jpg',route):
                path = store.root/'data/vsr'/route.lstrip('/')
                if path.stem not in store.dataset['images'] or not path.is_file():
                    return self.send(404,{'error':'Image not in this dataset.'})
                return self.send(200,path.read_bytes(),'image/jpeg')
            files = {'/':'index.html','/index.html':'index.html','/app.js':'app.js','/style.css':'style.css'}
            if route in files:
                path = store.root/'review'/files[route]
                content_type = {'html':'text/html; charset=utf-8','js':'text/javascript; charset=utf-8','css':'text/css; charset=utf-8'}[path.suffix[1:]]
                return self.send(200,path.read_bytes(),content_type)
            return self.send(404,{'error':'Not found.'})

        def do_POST(self):
            try:
                origin = self.headers.get('Origin')
                if origin and origin not in (f'http://127.0.0.1:{port}',f'http://localhost:{port}'):
                    return self.send(403,{'error':'Use the local review page.'})
                length = int(self.headers.get('Content-Length','0'))
                if not 0 < length <= 20000:
                    raise ReviewError('Invalid request size.')
                payload = json.loads(self.rfile.read(length))
                route = urlparse(self.path).path
                if route=='/api/decision':
                    result = store.save(payload)
                elif route=='/api/greenlight':
                    result = store.greenlight(payload)
                else:
                    return self.send(404,{'error':'Not found.'})
                return self.send(200,result)
            except (ReviewError,ValueError,KeyError) as error:
                return self.send(400,{'error':str(error)})
            except OSError as error:
                return self.send(500,{'error':f'Could not save locally: {error}'})
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8766)
    parser.add_argument('--root',type=Path,default=ROOT,help=argparse.SUPPRESS)
    args = parser.parse_args()
    store = ReviewStore(args.root)
    server = ThreadingHTTPServer(('127.0.0.1',args.port),handler_class(store,args.port))
    print(f'VSR review ready: http://127.0.0.1:{args.port}/',flush=True)
    print(f'Decisions save to {store.path}',flush=True)
    server.serve_forever()


if __name__=='__main__':
    main()
