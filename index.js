import * as dotenv from "dotenv";
import { Pinecone } from '@pinecone-database/pinecone';
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { OpenAIEmbeddings } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";

dotenv.config();
const pc = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY,
});
const index = pc.index("mypineconeindex");

(async () => {
    //1 - Create Pinecone index
    const existingIndexs = await pc.listIndexes();
    console.log(existingIndexs)
    if (!existingIndexs.indexes?.find(ele => ele.name === 'mypineconeindex')) {
        const createPineconeIndex = await pc.createIndex({
            name: "mypineconeindex",
            dimension: 1536,
            metric: 'cosine', 
            spec: { 
              serverless: { 
                cloud: 'aws', 
                region: 'us-east-1' 
              }
            } 
          });
        console.log("Created pinecone index", createPineconeIndex);
        //it takes a while to complete index creation. You could optionally wait for 1min here
    } else {
        console.log(`Index with name 'mypineconeindex' already exists`)
    }
    
    //2 - Update pinecone with document embeddings
    /* Load all PDFs within the specified directory */
    const directoryLoader = new DirectoryLoader("./documents", {
        ".pdf": (path) => new PDFLoader(path),
    });
    const docs = await directoryLoader.load();
    /**
     * docs contain combined array of pages from all the documents
     * Ex if 1 PDF has 3 pages and 1 PDF had 1 page, docs array has 4 pages
     */

    //From each page/doc item from docs array, split into chunks of 1000 characters
    const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 });
    for (let doc of docs) {
        /**
         * Each doc is broken into chunks. Here we have 4 chunks
         * Each chunk is an array of different length (here 3,3,3,7)
         * Each chunk contains multiple "Document" object. 
         * Each Document contains pageContent objectc containing segment of PDF data
         * */
        const chunks = await textSplitter.createDocuments([doc.pageContent])
        /**
         * For each of the chunk, create vector embedding one by one
         * Before embedding, replace new line with empty space since new lines can cause models to treat parts of text separately. 
         * Removing this helps with embedding quality
         */
        const cleanChunk = chunks.map((chunk) => chunk.pageContent.replace(/\n/g, " "));
        let embeddingArray = await new OpenAIEmbeddings().embedDocuments(cleanChunk);

        /** 
         * embeddingArray now replaced all contents inside each chunk into arrays with numbers
         * Here 4 arrays of length 3,3,3,7 as before. 
         * Each array containing sub-array of vector embeddings, 
         * successfully replacing the "pageContent" objects with clean vector embedded format
         *   [0.009098978, -0.0124089345,    0.0066301306,   -0.04462313,....]
             [-0.017448787,   0.032772664,   -0.0008432389,  -0.008601802,....]
             [-0.036368668,  -0.013668898,     0.022011897,   0.009732365,....]
             [-0.005407626,   0.023578338,   -0.0030409382,  0.0055744858,....]
         */

        /**
         * However, pinecone DB expects each chunk to be an object
         * Object to contain a unique id, values with the actual vector and optional metadata
         */
        const upsertValues = embeddingArray.map((item, i) => ({
            id: `doc-${doc.id || Date.now()}-${i}`,
            values: item,
            metadata: { text: cleanChunk[i], source: "pdf", pageIndex: i }
        }))

        /**store the  upsertValues using pinecone's upsert()
         * Loop through data in chunks and upsert in batches
        */
        const batchSize = 200;
        for (let i = 0; i < upsertValues.length; i += batchSize) {
            const batch = upsertValues.slice(i, i + batchSize);
            console.log("Upserting batch", batch, i)
            await index.upsert(batch);
        }
        //console.log("Data is stored!", storeData)
    }   

    //3 - Query vector database store and query LLM for answer
    let question = "Where all did the puppy travel to?";
    const queryEmbedding = await new OpenAIEmbeddings().embedQuery(question);

    /**Response structure:
     * Query results: {
        matches: [
            {
            id: 'doc-1742604060769-3',
            score: 0.729430735,
            values: [],
            sparseValues: undefined,
            metadata: [Object]
            },
            {
            id: 'doc-1742604060769-2',
            score: 0.713927269,
            values: [],
            sparseValues: undefined,
            metadata: [Object]
            },
            ...]
    Its all gibberish*/
    const queryResponse = await index.query({
            vector: queryEmbedding,
            topK: 10,               // return top 10 similar vectors
            includeMetadata: true,  // include metadata so you can see additional info
            includeValues: false    // optionally include the vector values if needed
    });
    console.log("Query results:", queryResponse);
    const context = queryResponse.matches.map(match => match.metadata.text || "").join("\n");

    /**Since the response had 'matches' array but everything in vector format, you need to use OpenAI API to convert them back to humam-readable format */
    const llm = new ChatOpenAI();
    const promptTemplate = `You are an assistant that summarizes documents.
    Context: {context}
    Question: {question}
    Answer:`;

    const prompt = ChatPromptTemplate.fromTemplate(promptTemplate);
    const chain = prompt.pipe(llm);
    const humanReadableResult =  await chain.invoke({context, question});

    console.log("Answer:", humanReadableResult);

})();
