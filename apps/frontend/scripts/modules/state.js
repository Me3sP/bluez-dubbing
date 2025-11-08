export const state = {
  options: null,
  asrModels: [],
  translationModels: [],
  ttsModels: [],
  uploadToken: "",
  sourceDescriptor: "",
  runId: "",
  targetLangs: [],
  latestResult: null,
  involveMode: false,
  pendingReviews: {
    transcription: null,
    alignment: null,
    tts: null
  },
  objectUrls: {
    source: null,
    result: null
  }
};
