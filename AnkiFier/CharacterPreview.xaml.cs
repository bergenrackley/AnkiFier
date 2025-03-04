using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Net.Http.Headers;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using static System.Net.Mime.MediaTypeNames;
using Rectangle = System.Windows.Shapes.Rectangle;
using Brushes = System.Windows.Media.Brushes;
using MessageBox = System.Windows.MessageBox;
using Accord.MachineLearning;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.CV;
using Newtonsoft.Json;
using Azure;
using Emgu.CV.Dai;
using OpenAI.Chat;
using static System.Runtime.InteropServices.JavaScript.JSType;
using static System.Windows.Forms.VisualStyles.VisualStyleElement.ToolTip;
using System.Security.Policy;
using System.Windows.Forms;
using Azure.AI.OpenAI;
using String = System.String;
using System.Text.Json;
using System.Collections;
using System.Data.Common;
using System.Drawing;
using System.Configuration;

namespace AnkiFier
{
    public partial class CharacterPreview : Window
    {
        private string AzureOpenAISubscriptionKey;
        private string AzureOpenAIEndpoint;
        private string AzureOpenAIModel;

        private List<Vocabulary> vocabulary = new List<Vocabulary>();

        public CharacterPreview(string imageFilePath, List<JObject> lines)
        {
            AzureOpenAISubscriptionKey = ConfigurationManager.AppSettings["AzureOpenAISubscriptionKey"].ToString();
            AzureOpenAIEndpoint = ConfigurationManager.AppSettings["AzureOpenAIEndpoint"].ToString();
            AzureOpenAIModel = ConfigurationManager.AppSettings["AzureOpenAIModel"].ToString();
            InitializeComponent();

            Status.Text = "Grouping...";
            List<System.Drawing.Rectangle> cells = DetectEdgesOfBlueCells(imageFilePath);

            GroupTextByCells(imageFilePath, lines, cells);
        }

        public CharacterPreview(string jsonPath)
        {
            InitializeComponent();
            try
            {
                Status.Text = "Loaded from JSON!";
                string jsonText = System.IO.File.ReadAllText(jsonPath);
                vocabulary = JsonConvert.DeserializeObject<List<Vocabulary>>(jsonText);
                VocabularyGrid.ItemsSource = null;
                VocabularyGrid.ItemsSource = vocabulary;
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Information);
                this.Close();
            }
        }

        private void GroupTextByCells(string imageFilePath, List<JObject> lines, List<System.Drawing.Rectangle> cells)
        {
            var bitmap = new BitmapImage(new Uri(imageFilePath, UriKind.RelativeOrAbsolute));
            
            List<Double> columns = KMeansClustering(cells.Select(x => Convert.ToDouble(x.X)).ToList(), 4);

            List<List<System.Drawing.Rectangle>> groupedCells = columns
                .Select(col => cells
                    .Where(cell => columns.MinBy(c => Math.Abs(c - Convert.ToDouble(cell.X))) == col)
                    .OrderBy(cell => Convert.ToDouble(cell.Y)) // Sort by Y value
                    .ToList()
                ).ToList();

            foreach (var column in groupedCells.Select((value, i) => new { i, value }))
            {
                List<System.Drawing.Rectangle> inserts = new List<System.Drawing.Rectangle>();
                foreach (var cell in column.value.Select((value, i) => new { i, value }))
                {
                    if (cell.i == 0) inserts.Add(new System.Drawing.Rectangle(Convert.ToInt32(columns[column.i]), 0, cell.value.Width, cell.value.Y));
                    int endy = (cell.i + 1 < column.value.Count) ? column.value[cell.i + 1].Y : Convert.ToInt32(bitmap.Height);
                    inserts.Add(new System.Drawing.Rectangle(Convert.ToInt32(columns[column.i]), cell.value.Y + cell.value.Height, cell.value.Width, endy - (cell.value.Y + cell.value.Height)));
                }
                cells.InsertRange(0, inserts);
            }

            List<List<string>> grammars = new List<List<string>>();

            cells.ForEach(cell =>
            {
                List<string> grammar = new List<string>();

                lines.ForEach(line =>
                {
                    List<int> boundingBox = line["boundingBox"].ToList().Select(x => int.Parse(x.ToString())).ToList();
                    //var (topLeft, width, height, angle) = GetRotatedBoundingBox(boundingBox);
                    if (boundingBox[0] > cell.X && boundingBox[0] < cell.X + cell.Width && boundingBox[1] > cell.Y && boundingBox[1] < cell.Y + cell.Height) grammar.Add(line["text"].ToString());
                });
                if (grammar.Any()) grammars.Add(grammar);
            });

            string json = JsonConvert.SerializeObject(grammars, Formatting.Indented);
            System.IO.File.WriteAllText("output.json", json);

            Status.Text = "Cleaning and classifying...";
            OpenAIMessage(json);
        }
        static List<System.Drawing.Rectangle> DetectEdgesOfBlueCells(string imagePath)
        {
            Mat image = CvInvoke.Imread(imagePath, ImreadModes.Color);
            if (image == null || image.IsEmpty)
            {
                Console.WriteLine("Error loading image.");
                return null;
            }

            MCvScalar lowerBound = new MCvScalar(252 - 10, 244 - 10, 226 - 10);
            MCvScalar upperBound = new MCvScalar(252 + 10, 244 + 10, 226 + 10);
            Mat mask = new Mat();
            CvInvoke.InRange(image, new ScalarArray(lowerBound), new ScalarArray(upperBound), mask);

            Mat processedImage = new Mat(image.Size, DepthType.Cv8U, 3);
            processedImage.SetTo(new MCvScalar(255, 255, 255)); // Set background to white
            image.CopyTo(processedImage, mask);  // Apply mask
            processedImage.SetTo(new MCvScalar(0, 0, 0), mask);  // Turn detected color to black

            Mat gray = new Mat();
            CvInvoke.CvtColor(processedImage, gray, ColorConversion.Bgr2Gray);

            CvInvoke.GaussianBlur(gray, gray, new System.Drawing.Size(5, 5), 0);

            Mat edges = new Mat();
            CvInvoke.Canny(gray, edges, 50, 150);

            // Step 5: Find Contours
            VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
            Mat hierarchy = new Mat();
            CvInvoke.FindContours(edges, contours, hierarchy, RetrType.External, ChainApproxMethod.ChainApproxSimple);


            List<System.Drawing.Rectangle> cells = new List<System.Drawing.Rectangle>();
            for (int i = 0; i < contours.Size; i++)
            {
                System.Drawing.Rectangle bbox = CvInvoke.BoundingRectangle(contours[i]);
                if (bbox.Width > 50 && bbox.Height > 20)  // Filtering small noise
                {
                    cells.Add(bbox);
                }
            }
            return cells;
        }

        private List<Double> KMeansClustering(List<Double> xPositions, int k)
        {

            double[][] data = xPositions.Select(x => new double[] { x }).ToArray();

            // Run K-Means clustering
            KMeans kmeans = new KMeans(k);
            KMeansClusterCollection clusters = kmeans.Learn(data);
            int[] labels = clusters.Decide(data); // Assigns each point to a cluster

            // Group the X positions by their assigned clusters
            var grouped = xPositions
                .Select((x, index) => new { X = x, Cluster = labels[index] })
                .GroupBy(item => item.Cluster)
                .OrderBy(group => group.Average(g => g.X)) // Ensure columns are in order
                .ToList();

            // Get the average X value for each cluster (column positions)
            List<Double> columnXPositions = grouped.Select(g => g.Average(i => i.X)).ToList();

            return columnXPositions;
        }

        public static (System.Drawing.Point topLeft, double width, double height, double angle) GetRotatedBoundingBox(List<int> boundingBox)
        {
            if (boundingBox.Count != 8)
                throw new ArgumentException("Bounding box must contain exactly 8 values (4 points).");

            // Extract points
            System.Drawing.Point topLeft = new System.Drawing.Point(boundingBox[0], boundingBox[1]);
            System.Drawing.Point topRight = new System.Drawing.Point(boundingBox[2], boundingBox[3]);
            System.Drawing.Point bottomRight = new System.Drawing.Point(boundingBox[4], boundingBox[5]);
            System.Drawing.Point bottomLeft = new System.Drawing.Point(boundingBox[6], boundingBox[7]);

            // Calculate width and height
            double width = Math.Sqrt(Math.Pow(topRight.X - topLeft.X, 2) + Math.Pow(topRight.Y - topLeft.Y, 2));
            double height = Math.Sqrt(Math.Pow(bottomLeft.X - topLeft.X, 2) + Math.Pow(bottomLeft.Y - topLeft.Y, 2));

            // Calculate angle (in degrees)
            double angle = Math.Atan2(topRight.Y - topLeft.Y, topRight.X - topLeft.X) * (180 / Math.PI);

            return (topLeft, width, height, angle);
        }

        private async Task OpenAIMessage(string message)
        {
            // Retrieve the OpenAI endpoint from environment variables
            if (string.IsNullOrEmpty(AzureOpenAIEndpoint))
            {
                Console.WriteLine("Please set the AZURE_OPENAI_ENDPOINT environment variable.");
                return;
            }

            if (string.IsNullOrEmpty(AzureOpenAISubscriptionKey))
            {
                Console.WriteLine("Please set the AZURE_OPENAI_KEY environment variable.");
                return;
            }

            AzureKeyCredential credential = new AzureKeyCredential(AzureOpenAISubscriptionKey);

            // Initialize the AzureOpenAIClient
            AzureOpenAIClient azureClient = new(new Uri(AzureOpenAIEndpoint), credential);

            // Initialize the ChatClient with the specified deployment name
            ChatClient chatClient = azureClient.GetChatClient(AzureOpenAIModel);

            // Create a list of chat messages
            var messages = new List<ChatMessage>
            {
                new SystemChatMessage("Extract structured vocabulary data from the given list. Each valid entry contains Japanese text, its romaji, and its English definition.\r\nReturn the results in a JSON array with the fields:\r\n'kanji': The standard kanji representation of the word, if applicable. If no kanji form exists, return an empty string.\r\n'japanese': The original Japanese text as provided. Do not modify this.\r\n'romaji': The romaji reading.\r\n'definition': The English definition.\r\nPreserve the original Japanese text exactly as given while separately providing the kanji form if possible. Fix fragmented text, remove numbering, and ignore non-relevant data while maintaining meaning.\r\nAlways use the kanji field when possible. Only leave 'kanji' empty when no kanji form exists. Fix fragmented text, remove numbering, and ignore non-relevant data while preserving meaning."),
            };

            messages.Add(new UserChatMessage(message));
            
            var options = new ChatCompletionOptions
            {
                Temperature = (float)0.0,
                MaxOutputTokenCount = 5000,

                TopP = (float)0.95,
                FrequencyPenalty = (float)0,
                PresencePenalty = (float)0
            };

            try
            {
                // Create the chat completion request
                ChatCompletion completion = await chatClient.CompleteChatAsync(messages, options);

                // Print the response
                if (completion != null)
                {
                    System.IO.File.WriteAllText("rawoutput.json", System.Text.Json.JsonSerializer.Serialize(completion, new JsonSerializerOptions() { WriteIndented = true }));;
                    System.IO.File.WriteAllText("content.json", System.Text.Json.JsonSerializer.Serialize(completion.Content[0].Text.Trim(), new JsonSerializerOptions() { WriteIndented = true }));;

                    string rawResponse = completion.Content[0].Text.Trim();

                    if (!rawResponse.StartsWith("{") && !rawResponse.StartsWith("["))
                    {
                        int jsonStart = rawResponse.IndexOf("{") < rawResponse.IndexOf("[") ? rawResponse.IndexOf("{") : rawResponse.IndexOf("[");
                        if (jsonStart != -1) rawResponse = rawResponse.Substring(jsonStart);
                    }

                    if (!rawResponse.EndsWith("}") && !rawResponse.EndsWith("]"))
                    {
                        int jsonEnd = rawResponse.LastIndexOf("}") > rawResponse.LastIndexOf("]") ? rawResponse.LastIndexOf("}") : rawResponse.LastIndexOf("]");
                        if (jsonEnd != -1) rawResponse = rawResponse.Substring(0, jsonEnd + 2);
                    }

                    // Try parsing the cleaned JSON
                    try
                    {
                        JArray jsonArray = JArray.Parse(rawResponse);
                        System.IO.File.WriteAllText("cleanedoutput.json", jsonArray.ToString(Formatting.Indented));
                        //MessageBox.Show(jsonArray.ToString(), "Parsed JSON Response");

                        vocabulary = jsonArray.ToObject<List<Vocabulary>>();
                        VocabularyGrid.ItemsSource = null;
                        VocabularyGrid.ItemsSource = vocabulary;
                        Status.Text = "Done!";

                    }
                    catch (JsonReaderException ex)
                    {
                        Trace.TraceError($"Invalid JSON: {ex.Message}\nResponse: {rawResponse}");
                    }
                }
                else
                {
                    Trace.TraceError("No response received.");
                }
            }
            catch (Exception ex)
            {
                Trace.TraceError($"An error occurred: {ex.Message}");
            }
        }

        private async void BoundingBox_Click(object sender, System.Windows.Input.MouseButtonEventArgs e)
        {
            if (sender is Rectangle rect && rect.Tag is string detectedText)
            {
                JObject wanikani = await GetWaniKani(detectedText, "kanji");
                if (wanikani != new JObject() && (wanikani["data"] as JArray).Any())
                {

                    List<String> radicals = new List<String>();
                    foreach (int id in wanikani["data"][0]["data"]["component_subject_ids"] as JArray)
                    {
                        JObject radical = await GetWaniKaniById(id);
                        Trace.TraceInformation(radical.ToString());
                        radicals.Add(radical["data"]["characters"].ToString() ?? radical["data"]["character_images"][0]["url"].ToString());
                    }

                    MessageBox.Show($"{String.Join(", ", wanikani["data"][0]["data"]["meanings"].Select(e => e["meaning"]).ToList())}\n{String.Join(", ", radicals)}\n{wanikani["data"][0]["data"]["meaning_mnemonic"]}\n{wanikani["data"][0]["data"]["reading_mnemonic"]}", "Bounding Box Clicked", MessageBoxButton.OK, MessageBoxImage.Information);
                }
            }
        }

        private async Task<JObject> GetWaniKani(string slug, string type)
        {
            string bearerToken = "c2653456-1c1d-4c02-840a-35e1d45e8ef8";
            using HttpClient client = new HttpClient();
            client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", bearerToken);
            HttpResponseMessage response = await client.GetAsync($"https://api.wanikani.com/v2/subjects?slugs={slug.ToLower()}&types={type}");

            if (response.IsSuccessStatusCode)
            {
                string jsonResponse = await response.Content.ReadAsStringAsync();
                return JObject.Parse(jsonResponse);
            }
            else
            {
                Trace.TraceError($"Failed to fetch data. Status code: {response.StatusCode}");
                return new JObject();
            }
        }

        private async Task<JObject> GetWaniKaniById(int id)
        {
            string bearerToken = "c2653456-1c1d-4c02-840a-35e1d45e8ef8";
            using HttpClient client = new HttpClient();
            client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", bearerToken);
            HttpResponseMessage response = await client.GetAsync($"https://api.wanikani.com/v2/subjects/{id}");

            if (response.IsSuccessStatusCode)
            {
                string jsonResponse = await response.Content.ReadAsStringAsync();
                return JObject.Parse(jsonResponse);
            }
            else
            {
                Trace.TraceError($"Failed to fetch data. Status code: {response.StatusCode}");
                return new JObject();
            }
        }

        private void SaveJson_OnClick(object sender, RoutedEventArgs e)
        {
            Microsoft.Win32.SaveFileDialog dlg = new Microsoft.Win32.SaveFileDialog();
            dlg.FileName = $"VocabularyJson {DateTime.Now.ToString("MM-dd-yyyy-HH-mm")}"; // Default file name
            dlg.DefaultExt = ".json"; // Default file extension
            dlg.Filter = "Json (.json)|*.json"; // Filter files by extension

            // Show save file dialog box
            Nullable<bool> result = dlg.ShowDialog();

            // Process save file dialog box results
            if (result == true)
            {
                // Save document
                string filename = dlg.FileName;

                string json = System.Text.Json.JsonSerializer.Serialize(VocabularyGrid.ItemsSource, new JsonSerializerOptions { WriteIndented = true });
                System.IO.File.WriteAllText(filename, json);

                MessageBox.Show("Vocab list saved successfully!", "Success", MessageBoxButton.OK, MessageBoxImage.Information);
                this.Close();
            }
        }

        private void SaveCSV_OnClick(object sender, RoutedEventArgs e)
        {
            Microsoft.Win32.SaveFileDialog dlg = new Microsoft.Win32.SaveFileDialog();
            dlg.FileName = $"VocabularyCSV {DateTime.Now.ToString("MM-dd-yyyy-HH-mm")}"; // Default file name
            dlg.DefaultExt = ".csv"; // Default file extension
            dlg.Filter = "CSV (.csv)|*.csv"; // Filter files by extension

            // Show save file dialog box
            Nullable<bool> result = dlg.ShowDialog();

            // Process save file dialog box results
            if (result == true)
            {
                StringBuilder csv = new StringBuilder();

                // Add the header row
                //csv.AppendLine("Japanese,Kanji,Romaji,Definition");

                // Loop through each vocab item and add it as a CSV row
                foreach (Vocabulary vocab in VocabularyGrid.ItemsSource)
                {
                    csv.AppendLine($"{EscapeCsv(vocab.japanese)},{EscapeCsv(vocab.kanji)},{EscapeCsv(vocab.romaji)},{EscapeCsv(vocab.definition)}");
                }

                // Save document
                string filename = dlg.FileName;

                System.IO.File.WriteAllText(filename, csv.ToString(), Encoding.UTF8);

                MessageBox.Show("Vocab list saved successfully!", "Success", MessageBoxButton.OK, MessageBoxImage.Information);
                this.Close();
            }
        }

        private static string EscapeCsv(string field)
        {
            if (string.IsNullOrEmpty(field)) return "";

            // Escape double quotes by replacing " with ""
            if (field.Contains(",") || field.Contains("\"") || field.Contains("\n"))
            {
                field = $"\"{field.Replace("\"", "\"\"")}\"";
            }

            return field;
        }

    }
}