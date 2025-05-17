import { useDropzone } from 'react-dropzone'
import { useCallback } from 'react'

function Sidebar ({ isOpen, onClose }) {
  const onDrop = useCallback(acceptedFiles => {
    console.log(acceptedFiles)
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop })

  return (
    <div
      className={`fixed left-0 top-0 w-64 h-full bg-gray-900 text-white transition-transform ${
        isOpen ? 'translate-x-0' : '-translate-x-full'
      }`}
    >
      <button onClick={onClose} className='absolute top-4 right-4 text-white'>
        X
      </button>
      <div className='p-4 my-12'>
        <div
          {...getRootProps()}
          className='border-2 border-dashed border-gray-600 p-4 text-center'
        >
          <input {...getInputProps()} />
          {isDragActive ? (
            <p>Drop the files here ...</p>
          ) : (
            <p>Drag 'n' drop some files here, or click to select files</p>
          )}
          <button className='mt-2 bg-blue-600 text-white px-4 py-2 rounded'>
            Browse File
          </button>
        </div>
        <button className='mt-4 bg-blue-600 text-white px-4 py-2 rounded w-full'>
          Process
        </button>
      </div>
    </div>
  )
}

export default Sidebar
