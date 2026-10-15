# Validation and remaining checks

## Checked during authoring

- Dataset checker passed: 60 training, 20 development and 20 test records; balanced categories; distinct input strings and IDs across splits; valid target schemas, routing and missing-field labels.
- All eight dependency-free scorer unit tests passed, including malformed JSON, duplicate keys, invented departments, incomplete missing-field lists, extra approval fields and incorrect routing.
- All Python scripts passed compilation checks.
- The HTML guide has unique IDs, valid internal navigation targets and existing local file links. It loads no external scripts, stylesheets, fonts, images or analytics.
- Embedded JavaScript passed Node syntax validation.
- compare.py successfully generated an HTML report from mock baseline/adapter inputs. Mock inputs and report are not included in the kit and are not model evaluation results.

## Not claimed

- This authoring session could not access /dev/nvidia0, so the supplied training, generation, peak memory, adapter reload and end-to-end GPU evaluation have not been executed here.
- Python package installation, connected model staging and disconnected wheel installation must be verified on the teaching environment before class.
- Browser layout and copy/print interactions have not been exercised with automated browser testing. The page uses responsive CSS, semantic navigation and native HTML controls; HTML and JavaScript received static checks.
- No trained model performance, classroom acceptance score or runtime duration is asserted in the kit.
- The original small LoRA exercise previously succeeded on the user's GPU; this larger structured-record exercise is a new run and needs its own evidence.

## Instructor completion check

1. Stage packages and model files. Verify the transferred inventory on a matching destination.
2. Run the guide with the network disconnected; confirm local model loading succeeds.
3. Run the baseline, training and development evaluation on the teaching GPU.
4. Fix a configuration using development evidence, then run the held-out final comparison.
5. Inspect the five embedded-instruction cases and ambiguous records individually.
6. Open the guide and generated report on the projector/browser used for class. Test copying commands and printing.
7. Complete INSTRUCTOR_NOTES.md with actual results, limitations and chosen run.

These checks are a teaching preparation process, not plant software qualification or approval.
