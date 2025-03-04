using System.Net.Http.Headers;
using System.Net.Http;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Windows.Media.Animation;
using System.Diagnostics;
using Newtonsoft.Json.Linq;
using System.Formats.Tar;
using System.Text.Json;
using System.Net;
using System.Configuration;

namespace AnkiFier
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private string ComputerVisionSubscriptionKey;
        private string ComputerVisionEndpoint;

        public MainWindow()
        {
            ComputerVisionSubscriptionKey = ConfigurationManager.AppSettings["ComputerVisionSubscriptionKey"].ToString();
            ComputerVisionEndpoint = ConfigurationManager.AppSettings["ComputerVisionEndpoint"].ToString();
            InitializeComponent();
        }

        private async void BeginOCR()
        {
            JObject text = await OCR();

            List<JObject> lines = (text["analyzeResult"]["readResults"][0]["lines"]).ToObject<List<JObject>>();

            CharacterPreview preview = new CharacterPreview(FileLocation.Text, lines);
            preview.Show();
        }

        private async Task<JObject> OCR()
        {
            using var client = new HttpClient();
            client.DefaultRequestHeaders.Add("Ocp-Apim-Subscription-Key", ComputerVisionSubscriptionKey);

            var requestUri = $"{ComputerVisionEndpoint}/vision/v3.2/read/analyze";
            var imageBytes = await System.IO.File.ReadAllBytesAsync(FileLocation.Text);
            using var content = new ByteArrayContent(imageBytes);
            content.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("application/octet-stream");

            var response = await client.PostAsync(requestUri, content);
            if (!response.IsSuccessStatusCode) return null;

            var operationUrl = response.Headers.GetValues("Operation-Location").FirstOrDefault();
            if (string.IsNullOrEmpty(operationUrl)) return null;

            // Wait for processing
            await Task.Delay(2000);

            while (true)
            {
                var resultResponse = await client.GetAsync(operationUrl);
                var resultString = await resultResponse.Content.ReadAsStringAsync();
                var resultJson = JObject.Parse(resultString);

                if (resultString.Contains("\"status\":\"succeeded\""))
                {
                    return resultJson;
                }
                await Task.Delay(1000);
            }
        }

        private void PickImage_OnClick(object sender, RoutedEventArgs e)
        {
            var fileDialog = new System.Windows.Forms.OpenFileDialog();
            var result = fileDialog.ShowDialog();
            switch (result)
            {
                case System.Windows.Forms.DialogResult.OK:
                    var file = fileDialog.FileName;
                    FileLocation.Text = file;
                    break;
                case System.Windows.Forms.DialogResult.Cancel:
                default:
                    FileLocation.Text = null;
                    break;
            }
        }

        private void ScanImage_OnClick(object sender, RoutedEventArgs e)
        {
            if (!String.IsNullOrEmpty(FileLocation.Text)) BeginOCR();
        }

        private void OpenJson_OnClick(object sender, RoutedEventArgs e)
        {
            var fileDialog = new System.Windows.Forms.OpenFileDialog();
            fileDialog.DefaultExt = ".json"; // Default file extension
            fileDialog.Filter = "Json (.json)|*.json"; // Filter files by extension
            var result = fileDialog.ShowDialog();
            switch (result)
            {
                case System.Windows.Forms.DialogResult.OK:
                    var file = fileDialog.FileName;
                    CharacterPreview preview = new CharacterPreview(file);
                    preview.ShowDialog();
                    break;
                case System.Windows.Forms.DialogResult.Cancel:
                default:
                    FileLocation.Text = null;
                    break;
            }
        }
    }
}