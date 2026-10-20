"""Exercise real SDK serialization against a local HTTP mock, never a paid API."""
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import api

class Handler(BaseHTTPRequestHandler):
    calls=0
    def log_message(self,*args):pass
    def do_POST(self):
        body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        Handler.calls+=1
        messages=body.get('messages',body.get('input'))
        view=json.loads(messages[-1]['content'])
        if 'targets' in view:answer=dict(predictions=[dict(date=d['date'],entries=123) for d in view['targets']],reason='mock')
        elif 'person' in view:answer=dict(probabilities=[.6,.1,0,.1,.2],reason='mock')
        else:answer=dict(entries=123,reason='mock')
        if self.path.endswith('/responses'):
            response=dict(id='resp_mock',object='response',created_at=0,model=body['model'],status='completed',output=[dict(id='msg_mock',type='message',role='assistant',status='completed',content=[dict(type='output_text',text=json.dumps(answer),annotations=[])])],usage=dict(input_tokens=50,output_tokens=10,total_tokens=60))
        else:response=dict(id='chat_mock',object='chat.completion',created=0,model=body['model'],choices=[dict(index=0,message=dict(role='assistant',content=json.dumps(answer)),finish_reason='stop')],usage=dict(prompt_tokens=50,completion_tokens=10,total_tokens=60))
        raw=json.dumps(response).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)

class AdapterTest(unittest.TestCase):
    def test_chat_responses_cache_and_budget(self):
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as folder:
                api.LOGDIR=folder;api.LEDGER=[];api.CAP=100000;api.API_KEY='local';api.BASE_URL=f'http://127.0.0.1:{server.server_port}/v1';api.MODEL='mock';api.EXTRA_BODY={}
                for mode in ['chat','responses']:
                    api.API_MODE=mode
                    for route,view,person in [('direct',{'target':{}},None),('batch',{'targets':[{'date':'2023-10-07'}]},None),('role',{'person':{}},{'budget_usd':6,'mobility_limited':False})]:
                        first=api.request(route,view,person);before=Handler.calls
                        second=api.request(route,view,person,offline=True)
                        self.assertEqual(first,second);self.assertEqual(before,Handler.calls)
                self.assertEqual(Handler.calls,6)
                api.MODEL='different-model';api.CAP=0
                with self.assertRaisesRegex(RuntimeError,'budget'):api.request('direct',{'target':{}},None)
                self.assertEqual(Handler.calls,6)
        finally:server.shutdown();server.server_close()

if __name__=='__main__':unittest.main()
