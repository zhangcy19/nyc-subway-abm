"""Exercise real SDK serialization against a local HTTP mock, never a paid API."""
import json
import tempfile
import threading
import unittest
from datetime import datetime
from unittest.mock import patch
from contextlib import ExitStack
from types import SimpleNamespace
import httpx
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
    def test_request_window(self):
        window=dict(timezone='Asia/Shanghai',start='18:00',end='08:55')
        with patch.object(api,'REQUEST_WINDOW',window):
            for clock,allowed in [('17:59',False),('18:00',True),('23:59',True),('00:00',True),('08:54',True),('08:55',False),('09:00',False),('12:30',False)]:
                with self.subTest(clock=clock):
                    now=datetime.fromisoformat('2026-10-20T'+clock)
                    if allowed:api.check_request_window(now)
                    else:
                        with self.assertRaisesRegex(RuntimeError,'Outside configured API hours'):api.check_request_window(now)

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

class TransportRetryTest(unittest.TestCase):
    def setUp(self):
        stack=ExitStack();self.addCleanup(stack.close)
        folder=stack.enter_context(tempfile.TemporaryDirectory())
        settings=dict(LOGDIR=folder,LEDGER=[],CAP=100000,RESERVED=0,LAST=0.,NETWORK_FAILED=False,
                      REQUEST_WINDOW=None,API_KEY='local',BASE_URL='http://mock.invalid/v1',
                      MODEL='mock',API_MODE='responses',EXTRA_BODY={})
        for name,value in settings.items():stack.enter_context(patch.object(api,name,value))
        self.sleep=stack.enter_context(patch.object(api.time,'sleep'))
        self.client=stack.enter_context(patch.object(api,'OpenAI')).return_value
        self.answer=SimpleNamespace(output_text='{"entries":123,"reason":"mock"}',status='completed',id='resp_mock',
                                    usage=SimpleNamespace(model_dump=lambda:dict(input_tokens=50,output_tokens=10,total_tokens=60)))

    def connection_error(self):return api.APIConnectionError(request=httpx.Request('POST','http://mock.invalid/v1/responses'))

    def test_walking_projection_keeps_raw_reply_and_separate_cache(self):
        person=dict(budget_usd=15,mobility_limited=True)
        raw='{"probabilities":[0.42,0.28,0,0.05,0.25],"reason":"mock"}'
        self.answer.output_text=raw;self.client.responses.create.return_value=self.answer
        with patch.object(api,'WALKING_POLICY','zero_walk_renormalize'):
            got=api.request('role',{'person':person},person)
            self.assertTrue(api.np.allclose(got['probabilities'],[.42/.95,.28/.95,0,0,.25/.95]))
            self.assertEqual(got['probability_adjustment']['removed_walk_probability'],.05)
            self.assertEqual(api.ledger()[0]['raw'],raw)
            self.assertEqual(api.request('role',{'person':person},person,offline=True),got)
        with patch.object(api,'WALKING_POLICY','strict'):
            with self.assertRaisesRegex(RuntimeError,'Missing real API response'):api.request('role',{'person':person},person,offline=True)
        self.assertEqual(self.client.responses.create.call_count,1)
        self.assertEqual(self.client.responses.create.call_args.kwargs['input'][0]['content'],api.ROLE_SYSTEM)

    def test_walking_projection_preserves_other_validation(self):
        person=dict(budget_usd=15,mobility_limited=True)
        with patch.object(api,'WALKING_POLICY','strict'):
            with self.assertRaisesRegex(ValueError,'Walking unavailable'):api.validate(dict(probabilities=[.4,.2,0,.1,.3],reason='mock'),'role',person)
        with patch.object(api,'WALKING_POLICY','zero_walk_renormalize'):
            with self.assertRaisesRegex(ValueError,'No probability mass'):api.validate(dict(probabilities=[0,0,0,1,0],reason='mock'),'role',person)
            with self.assertRaisesRegex(ValueError,'Taxi unavailable'):api.validate(dict(probabilities=[.4,.1,.1,.1,.3],reason='mock'),'role',person)
            with self.assertRaisesRegex(ValueError,'sum to 1'):api.validate(dict(probabilities=[.4,.2,0,.1,.1],reason='mock'),'role',person)
            got=api.validate(dict(probabilities=[.4,.2,0,.1,.3],reason='mock'),'role',dict(budget_usd=15,mobility_limited=False))
            self.assertEqual(got['probabilities'],[.4,.2,0,.1,.3])
            self.assertNotIn('probability_adjustment',got)

    def test_connection_then_timeout_then_success_and_cache(self):
        timeout=api.APITimeoutError(request=httpx.Request('POST','http://mock.invalid/v1/responses'))
        self.client.responses.create.side_effect=[self.connection_error(),timeout,self.answer]
        first=api.request('direct',{'target':{}})
        self.assertEqual(first['entries'],123)
        self.assertEqual([r['status'] for r in api.ledger()],['error','error','valid'])
        self.assertEqual(api.RESERVED,0)
        self.assertFalse(api.NETWORK_FAILED)
        self.sleep.assert_any_call(5);self.sleep.assert_any_call(15)
        self.assertEqual(api.request('direct',{'target':{}},offline=True),first)
        self.assertEqual(self.client.responses.create.call_count,3)

    def test_unavailable_taxi_and_walk_projection(self):
        person=dict(budget_usd=15,mobility_limited=True)
        raw=dict(probabilities=[.55,.12,.03,.10,.20],reason='mock')
        with patch.object(api,'TAXI_POLICY','zero_taxi_renormalize'),patch.object(api,'WALKING_POLICY','zero_walk_renormalize'):
            got=api.validate(raw,'role',person)
            self.assertTrue(api.np.allclose(got['probabilities'],[.55/.87,.12/.87,0,0,.20/.87]))
            self.assertEqual(got['probability_adjustment']['original_probabilities'],raw['probabilities'])
            self.assertEqual(got['probability_adjustment']['removed_taxi_probability'],.03)
            with self.assertRaisesRegex(ValueError,'No probability mass'):
                api.validate(dict(probabilities=[0,0,.5,.5,0],reason='mock'),'role',person)
            affordable=api.validate(raw,'role',dict(budget_usd=20,mobility_limited=False))
            self.assertEqual(affordable,raw)
            with patch.object(api,'TAXI_POLICY','strict'):
                with self.assertRaisesRegex(ValueError,'Taxi unavailable'):api.validate(raw,'role',person)

    def test_three_failures_open_circuit(self):
        self.client.responses.create.side_effect=self.connection_error()
        with self.assertRaisesRegex(RuntimeError,'three unsuccessful attempts'):api.request('direct',{'target':{}})
        self.assertEqual(len(api.ledger()),3)
        self.assertTrue(api.NETWORK_FAILED)
        self.assertEqual(api.RESERVED,0)
        with self.assertRaisesRegex(RuntimeError,'Network circuit open'):api.request('direct',{'target':{'other':True}})
        self.assertEqual(self.client.responses.create.call_count,3)

    def test_retry_rechecks_budget(self):
        view={'target':{}}
        api.CAP=len((api.DIRECT_SYSTEM+json.dumps(view,separators=(',',':'))).encode())+api.MAX_OUTPUT+100
        self.client.responses.create.side_effect=self.connection_error()
        with self.assertRaisesRegex(RuntimeError,'budget'):api.request('direct',view)
        self.assertEqual(self.client.responses.create.call_count,1)
        self.assertEqual(api.RESERVED,0)

    def test_retry_rechecks_time_window(self):
        self.client.responses.create.side_effect=self.connection_error()
        with patch.object(api,'check_request_window',side_effect=[None,None,RuntimeError('Outside configured API hours')]):
            with self.assertRaisesRegex(RuntimeError,'Outside configured API hours'):api.request('direct',{'target':{}})
        self.assertEqual(self.client.responses.create.call_count,1)
        self.assertEqual(api.RESERVED,0)

if __name__=='__main__':unittest.main()
