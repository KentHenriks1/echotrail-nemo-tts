#!/usr/bin/env python3
"""Test Norwegian tokenizer options for NeMo TTS."""

def test_phonemizer():
    """Test phonemizer + espeak-ng for Norwegian."""
    try:
        from phonemizer import phonemize
        result = phonemize(
            'Hei, dette er en test av norsk tale.',
            language='nb', backend='espeak', strip=True
        )
        print(f"Norwegian IPA: {result}")
        result2 = phonemize(
            'Kjøkkenet har fine møbler og flotte bøker.',
            language='nb', backend='espeak', strip=True
        )
        print(f"Norwegian IPA2: {result2}")
        result3 = phonemize(
            'Været er fint i dag, og vi går en tur.',
            language='nb', backend='espeak', strip=True
        )
        print(f"Norwegian IPA3: {result3}")
        return True
    except ImportError as e:
        print(f"phonemizer not installed: {e}")
        return False
    except Exception as e:
        print(f"phonemizer error: {e}")
        return False

def test_nemo_ipa_g2p():
    """Test NeMo's IpaG2p with Norwegian locale."""
    try:
        from nemo.collections.tts.g2p.models.i18n_ipa import IpaG2p
        import inspect
        sig = inspect.signature(IpaG2p.__init__)
        print(f"IpaG2p params: {sig}")
        
        # Try Norwegian locale
        try:
            g2p = IpaG2p(phoneme_dict=None, locale="nb-NO")
            result = g2p("Hei dette er en test")
            print(f"nb-NO result: {result}")
        except Exception as e:
            print(f"nb-NO failed: {e}")
        
        # Try with just phonemizer backend
        try:
            g2p = IpaG2p(phoneme_dict=None, locale=None, use_stresses=False)
            print(f"IpaG2p created (no locale)")
        except Exception as e:
            print(f"IpaG2p no-locale failed: {e}")
            
    except ImportError as e:
        print(f"NeMo IpaG2p not available: {e}")
    except Exception as e:
        print(f"IpaG2p error: {e}")

def test_espeak_directly():
    """Test espeak-ng directly for Norwegian."""
    import subprocess
    try:
        result = subprocess.run(
            ['espeak-ng', '--ipa', '-v', 'nb', '-q', 'Hei dette er en test av norsk tale'],
            capture_output=True, text=True
        )
        print(f"espeak-ng Norwegian: {result.stdout.strip()}")
        result2 = subprocess.run(
            ['espeak-ng', '--ipa', '-v', 'nb', '-q',
             'Kjøkkenet har fine møbler og flotte bøker'],
            capture_output=True, text=True
        )
        print(f"espeak-ng Norwegian2: {result2.stdout.strip()}")
        result3 = subprocess.run(
            ['espeak-ng', '--voices=nb'], capture_output=True, text=True
        )
        print(f"espeak-ng nb voice: {result3.stdout.strip()}")
        return True
    except FileNotFoundError:
        print("espeak-ng not installed")
        return False

def test_nemo_tokenizer_with_ipa():
    """Test creating IPATokenizer with Norwegian phoneme set."""
    try:
        from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import IPATokenizer
        from nemo.collections.tts.g2p.models.i18n_ipa import IpaG2p
        
        # Create IPA G2P for Norwegian
        g2p = IpaG2p(
            phoneme_dict=None,
            locale="nb-NO",
            use_stresses=True,
        )
        tok = IPATokenizer(locale="nb-NO", g2p=g2p)
        print(f"IPATokenizer created with {len(tok.tokens)} tokens")
        
        # Test encoding
        encoded = tok.encode("Hei dette er en test")
        print(f"Encoded: {encoded}")
        return True
    except Exception as e:
        print(f"IPATokenizer failed: {e}")
        import traceback; traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Testing Norwegian tokenizer options ===")
    print()
    print("--- 1. espeak-ng direct ---")
    test_espeak_directly()
    print()
    print("--- 2. phonemizer ---")
    test_phonemizer()
    print()
    print("--- 3. NeMo IpaG2p ---")
    test_nemo_ipa_g2p()
    print()
    print("--- 4. NeMo IPATokenizer ---")
    test_nemo_tokenizer_with_ipa()
