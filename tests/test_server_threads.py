"""Model-free integration checks; pass TTS_SERVER as an absolute binary path."""
import os
from pathlib import Path
import subprocess
import unittest


class ServerThreadsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = Path(os.environ['TTS_SERVER']).resolve(strict=True)
        cls.env = {**os.environ, 'GGML_BACKEND': 'CPU', 'LD_LIBRARY_PATH': str(cls.server.parent)}

    def run_server(self, *args):
        return subprocess.run([str(self.server), *args], env=self.env, text=True,
                              capture_output=True, timeout=30)

    def test_help_documents_flag(self):
        result = self.run_server('--help')
        self.assertEqual(result.returncode, 0)
        self.assertIn('--threads <n>', result.stderr)

    def test_invalid_thread_values_are_rejected_before_loading(self):
        for args in (('--threads',), ('--threads', '0'), ('--threads', '-2'),
                     ('--threads', '1x'), ('--threads', '999999999999999999999')):
            with self.subTest(args=args):
                result = self.run_server(*args)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('--threads requires a positive integer', result.stderr)
                self.assertNotIn('[Load]', result.stderr)

    def test_omitted_threads_keep_the_original_default(self):
        result = self.run_server('--model', '/no-such-talker.gguf', '--codec', '/no-such-codec.gguf')
        self.assertNotEqual(result.returncode, 0)
        expected = max(1, (os.cpu_count() or 0) // 2)
        self.assertIn(f'CPU threads: {expected}', result.stderr)

    def test_requested_threads_reach_cpu_backend_without_model(self):
        result = self.run_server('--threads', '3', '--model', '/no-such-talker.gguf',
                                 '--codec', '/no-such-codec.gguf')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('CPU threads: 3', result.stderr)
        self.assertIn('/no-such-', result.stderr)


if __name__ == '__main__':
    unittest.main()
